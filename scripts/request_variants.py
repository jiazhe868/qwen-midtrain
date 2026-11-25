"""Compare wordings of the teacher's request on the same passages: which wording keeps the rewrites as varied as the
real passages they come from?

Why this script exists. The first two batches of rewrites failed the diversity gate (README Appendix D). The rewrites kept
their topics (92% of the parents' spread once each set's mean direction is removed) but shared one direction, the
register of talking about a text. Each wording below is sent for the same 150 passages, each passage in the one style
the v2 batch gave it, and every result is measured against the same passages: the Vendi ratio and bigram-entropy drop
of the gate, the length of the mean unit embedding (a shared register makes it long), and how many rewrites point at
"the passage", "the text", "the excerpt", "the source" or "the author" in their first 400 words. Run once on 11-21;
the wording that won became the v3 request of synthetic_rephrase.py.

    source scripts/env.sh
    $MIDTRAIN_PYTHON scripts/request_variants.py --endpoint http://127.0.0.1:8200 plain own_voice_specific ...
    $MIDTRAIN_PYTHON scripts/request_variants.py --measure-only v2_reader_voice plain keep_genre formats specific \
        own_voice own_voice_specific paraphrase          # results/synthetic_request_variants.json
    $MIDTRAIN_PYTHON scripts/request_variants.py --diagnose v1 v2 v3   # results/synthetic_shared_direction.json

`--diagnose` explains a batch's Vendi score: for 600 rewrites (one per parent) and their parents it gives the Vendi
score as the gate measures it and after removing each set's mean direction, and the length of the mean unit embedding;
for the plain wording it lists the words whose documents lie furthest along, and least far along, the direction from
the parents' mean to the rewrites' mean.
"""
import collections
import argparse
import concurrent.futures
import json
import random
import re
import urllib.request

import numpy as np

from midtrain.curation_funnel import default_embedding_model, embed_documents
from midtrain.data_hygiene import load_qwen_tokenize_function, measure_token_statistics, vendi_score_of_unit_vectors
from midtrain.data_sources import data_root, repository_root
from midtrain.synthetic_rephrase import read_json_lines, reason_to_drop, shared_instruction, style_instructions

variant_directory = data_root() / "synthetic" / "request_variants"
genre_of_source = {"wiki_en": "an encyclopedia", "textbooks": "a textbook", "papers": "a scientific paper",
                   "math_web": "a mathematics website", "forum_qa": "a question-and-answer forum"}
formats_of_style = {
    "explain_from_first_principles": ["an encyclopedia article", "a section of a textbook chapter",
                                      "the transcript of a recorded lecture", "a long magazine feature",
                                      "a blog post by a practitioner", "a technical memo to colleagues",
                                      "a letter answering a reader's question", "the script of an educational video"],
    "structured_notes": ["a one-page cheat sheet", "a set of flashcards with explanations on the back",
                         "a glossary with a worked example for each entry", "an exam study guide",
                         "the text of lecture slides with speaker notes", "a reference card for practitioners",
                         "an annotated outline"],
    "socratic_dialogue": ["a radio interview with an expert", "a conversation during office hours",
                          "two colleagues working at a whiteboard", "a podcast transcript", "a mentor guiding an intern",
                          "a parent and a teenager talking at the kitchen table", "a short play with stage directions"],
    "question_answer_derivation": ["an exam paper with model answers", "a page of frequently asked questions",
                                   "a forum thread with a question and an accepted answer", "a quiz with explained answers",
                                   "a job interview in which the candidate reasons aloud", "a reader-mail column",
                                   "a study group's review sheet"],
    "worked_problem": ["a homework solutions handout", "a competition problem write-up",
                       "a textbook box of worked examples", "a tutoring session transcript", "a lab notebook entry",
                       "an engineering calculation note", "a puzzle column with solutions"],
    "what_if_analysis": ["a design review", "a thought-experiment essay", "a structured debate between two people",
                         "a scenario-planning memo", "a referee report that questions the assumptions",
                         "museum exhibit text with what-if panels", "the transcript of a classroom discussion"],
}
own_voice_words = ("Write as the author of a document that presents this material in your own words. Never refer to a "
                   "passage, a text, a source, an excerpt or an author. ")
specific_words = "Keep the material's own names, places, dates, numbers and examples, and open with them. "
source_reference = re.compile(r"\b(the|this) (passage|text|excerpt|source|author)\b", re.IGNORECASE)


def request_text(passage: dict, style_name: str, variant: str, random_generator: random.Random) -> str:
    """The user message of one rewrite under one wording; "paraphrase" replaces the style instruction itself."""
    if variant == "paraphrase":
        return ("Rewrite the passage in your own words for the same kind of reader and in the same genre, keeping every "
                "name, number and example. Where the original skips a step of reasoning, add it. " + shared_instruction
                + "\n\nPassage:\n" + passage["passage"])
    prefix = {
        "plain": "",
        "keep_genre": (f"The passage comes from {genre_of_source[passage['parent_source']]}; keep that register and "
                       "write for the kind of reader it addresses. "),
        "formats": f"Present it as {random_generator.choice(formats_of_style[style_name])}. ",
        "specific": ("Stay with this passage's particular case: keep its own names, places, dates, numbers and "
                     "examples, open with them, and do not drift into general statements about the topic. "),
        "own_voice": own_voice_words,
        "own_voice_specific": own_voice_words + specific_words,
    }[variant]
    return f"{prefix}{style_instructions[style_name]} {shared_instruction}\n\nPassage:\n{passage['passage']}"


def ask_teacher_with_text(endpoint: str, message_text: str) -> tuple[str, str]:
    """The same request as synthetic_rephrase.ask_teacher, with the message given whole; temperature 1.0."""
    request_body = {"model": "teacher", "messages": [{"role": "user", "content": message_text}], "max_tokens": 1800,
                    "temperature": 1.0, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(f"{endpoint}/v1/chat/completions", json.dumps(request_body).encode("utf-8"),
                                     {"Content-Type": "application/json"})
    choice = json.load(urllib.request.urlopen(request, timeout=1800))["choices"][0]
    return (choice["message"].get("content") or "").strip(), choice.get("finish_reason") or ""


def generate_variant(endpoint: str, passages: list[dict], variant: str) -> None:
    """Write synthetic/request_variants/<variant>.jsonl: one rewrite per passage, all requests at once."""
    random_generator = random.Random(0)          # the same draw of formats on every run
    jobs = [(passage, passage["styles"][0]) for passage in passages]
    texts = [request_text(passage, style_name, variant, random_generator) for passage, style_name in jobs]

    def do_one(job_index: int) -> dict:
        passage, style_name = jobs[job_index]
        output_text, finish_reason = ask_teacher_with_text(endpoint, texts[job_index])
        return {"parent_doc_id": passage["parent_doc_id"], "parent_source": passage["parent_source"], "style": style_name,
                "variant": variant, "drop_reason": reason_to_drop(output_text, passage["passage"], finish_reason),
                "text": output_text}
    with concurrent.futures.ThreadPoolExecutor(len(jobs)) as request_pool:
        records = list(request_pool.map(do_one, range(len(jobs))))
    variant_directory.mkdir(parents=True, exist_ok=True)
    with (variant_directory / f"{variant}.jsonl").open("w") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def measure_variant(records: list[dict], passage_by_parent: dict, tokenize_texts) -> dict:
    """The gate's numbers for the kept rewrites of one wording, against exactly their own parents."""
    kept = [record for record in records if not record["drop_reason"]]
    texts = [record["text"] for record in kept]
    parent_texts = [passage_by_parent[record["parent_doc_id"]]["passage"] for record in kept]
    embeddings = embed_documents(texts, default_embedding_model, "cpu")
    parent_embeddings = embed_documents(parent_texts, default_embedding_model, "cpu")
    token_ids, parent_token_ids = tokenize_texts(texts), tokenize_texts(parent_texts)
    token_sample_size = int(0.95 * min(sum(map(len, token_ids)), sum(map(len, parent_token_ids))))
    bigram_entropy = measure_token_statistics(token_ids, token_sample_size)["bigram_entropy_bits"]
    parent_bigram_entropy = measure_token_statistics(parent_token_ids, token_sample_size)["bigram_entropy_bits"]
    vendi, parent_vendi = vendi_score_of_unit_vectors(embeddings), vendi_score_of_unit_vectors(parent_embeddings)
    return {"rewrites": len(records), "kept": len(kept), "vendi": vendi, "parent_vendi": parent_vendi,
            "vendi_ratio": vendi / parent_vendi, "mean_vector_length": float(np.linalg.norm(embeddings.mean(0))),
            "parent_mean_vector_length": float(np.linalg.norm(parent_embeddings.mean(0))),
            "bigram_entropy_drop_bits": parent_bigram_entropy - bigram_entropy, "token_sample_size": token_sample_size,
            "median_words": int(np.median([len(text.split()) for text in texts])),
            "rewrites_referring_to_their_source": sum(1 for text in texts if source_reference.search(" ".join(text.split()[:400])))}


def vendi_without_mean_direction(embeddings: np.ndarray) -> float:
    """The Vendi score after removing the set's mean direction: how spread the documents are around their centre."""
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    centered /= np.linalg.norm(centered, axis=1, keepdims=True)
    return vendi_score_of_unit_vectors(centered)


def diagnose_batches(batch_names: list[str]) -> dict:
    """Vendi score with and without the mean direction, and the mean-vector length, for each batch and its parents;
    then the words along the direction that separates the plain wording's rewrites from their parents."""
    diagnosis = {"embedding_model": default_embedding_model, "batches": {}}
    for batch_name in batch_names:
        passages = {passage["parent_doc_id"]: passage
                    for passage in read_json_lines(data_root() / "synthetic" / batch_name / "passages.jsonl")}
        records, parents_seen = [], set()
        for record in read_json_lines(data_root() / "synthetic" / batch_name / "rewrites.jsonl"):
            if record["drop_reason"] or record["parent_doc_id"] in parents_seen:
                continue
            parents_seen.add(record["parent_doc_id"])
            records.append(record)
            if len(records) == 600:
                break
        entry = {"documents": len(records)}
        for side_name, texts in (("parents", [passages[record["parent_doc_id"]]["passage"] for record in records]),
                                 ("rewrites", [record["text"] for record in records])):
            embeddings = embed_documents(texts, default_embedding_model, "cpu")
            entry[side_name] = {"vendi": vendi_score_of_unit_vectors(embeddings),
                                "vendi_without_mean_direction": vendi_without_mean_direction(embeddings),
                                "mean_vector_length": float(np.linalg.norm(embeddings.mean(axis=0)))}
        entry["spread_ratio_without_mean_direction"] = (entry["rewrites"]["vendi_without_mean_direction"]
                                                        / entry["parents"]["vendi_without_mean_direction"])
        diagnosis["batches"][batch_name] = entry
        print(f"{batch_name}: {len(records)} rewrites; vendi ratio {entry['rewrites']['vendi'] / entry['parents']['vendi']:.3f}, "
              f"without the mean direction {entry['spread_ratio_without_mean_direction']:.3f}; mean-vector length "
              f"{entry['parents']['mean_vector_length']:.3f} -> {entry['rewrites']['mean_vector_length']:.3f}", flush=True)
    # words along the shared direction, for the plain wording on the 150 test passages
    passages = {passage["parent_doc_id"]: passage for passage in read_json_lines(data_root() / "synthetic" / "v2" / "passages.jsonl")}
    records = [record for record in read_json_lines(variant_directory / "plain.jsonl") if not record["drop_reason"]]
    rewrite_texts = [record["text"] for record in records]
    parent_texts = [passages[record["parent_doc_id"]]["passage"] for record in records]
    rewrite_embeddings = embed_documents(rewrite_texts, default_embedding_model, "cpu")
    parent_embeddings = embed_documents(parent_texts, default_embedding_model, "cpu")
    direction = rewrite_embeddings.mean(axis=0) - parent_embeddings.mean(axis=0)
    direction /= np.linalg.norm(direction)
    projections = np.concatenate([rewrite_embeddings @ direction, parent_embeddings @ direction])
    documents_of_word = collections.defaultdict(list)
    for document_index, text in enumerate(rewrite_texts + parent_texts):
        # the embedding reads about the first 512 tokens, so only the first 400 words count
        for word in set(re.findall(r"[a-z]+", text.lower())[:400]):
            documents_of_word[word].append(document_index)
    word_scores = sorted((float(np.mean(projections[indices])), word, len(indices))
                         for word, indices in documents_of_word.items() if len(indices) >= 25)
    diagnosis["plain_wording_direction"] = {
        "documents": len(records) * 2,
        "projection_mean": {"rewrites": float(projections[:len(records)].mean()), "parents": float(projections[len(records):].mean())},
        "words_furthest_along": [{"word": word, "documents": count} for _, word, count in word_scores[-20:][::-1]],
        "words_least_far_along": [{"word": word, "documents": count} for _, word, count in word_scores[:20]]}
    print("words furthest along:", ", ".join(item["word"] for item in diagnosis["plain_wording_direction"]["words_furthest_along"]))
    print("words least far along:", ", ".join(item["word"] for item in diagnosis["plain_wording_direction"]["words_least_far_along"]))
    return diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variants", nargs="+", help="v2_reader_voice (the v2 batch's own rewrites) or a wording above")
    parser.add_argument("--endpoint", default=None, help="the teacher; needed for wordings not generated yet")
    parser.add_argument("--passage-count", type=int, default=150)
    parser.add_argument("--measure-only", action="store_true")
    parser.add_argument("--diagnose", action="store_true", help="the positional names are batches (v1 v2 ...)")
    arguments = parser.parse_args()
    if arguments.diagnose:
        output_path = repository_root / "results" / "synthetic_shared_direction.json"
        output_path.write_text(json.dumps(diagnose_batches(arguments.variants), indent=1) + "\n")
        print(f"wrote {output_path}")
        return
    passages = read_json_lines(data_root() / "synthetic" / "v2" / "passages.jsonl")[:arguments.passage_count]
    passage_by_parent = {passage["parent_doc_id"]: passage for passage in passages}
    v2_records = {record["parent_doc_id"]: record for record in read_json_lines(data_root() / "synthetic" / "v2" / "rewrites.jsonl")}
    tokenize_texts = load_qwen_tokenize_function()
    results = {"passages": arguments.passage_count, "embedding_model": default_embedding_model, "variants": {}}
    for variant in arguments.variants:
        if variant == "v2_reader_voice":
            records = [v2_records[parent] for parent in passage_by_parent if parent in v2_records]
        else:
            if not (variant_directory / f"{variant}.jsonl").exists():
                assert not arguments.measure_only and arguments.endpoint, f"{variant} is not generated yet: give --endpoint"
                generate_variant(arguments.endpoint, passages, variant)
            records = read_json_lines(variant_directory / f"{variant}.jsonl")
        results["variants"][variant] = measure_variant(records, passage_by_parent, tokenize_texts)
        entry = results["variants"][variant]
        print(f"{variant:20s} kept {entry['kept']:3d}  vendi ratio {entry['vendi_ratio']:.3f}  mean-vector length "
              f"{entry['mean_vector_length']:.3f} (parents {entry['parent_mean_vector_length']:.3f})  bigram drop "
              f"{entry['bigram_entropy_drop_bits']:.2f}  refer to their source {entry['rewrites_referring_to_their_source']}",
              flush=True)
    output_path = repository_root / "results" / "synthetic_request_variants.json"
    output_path.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
