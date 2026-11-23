"""Tests of the teacher rewriting (synthetic_rephrase.py) that need no GPU and no teacher; seconds.

Why these tests exist. The first two batches of rewrites failed the diversity gate, and the main cause was in the
wording of the request: 58% of the rewrites in a 150-passage test pointed their reader at "the passage" or "the text",
a source that reader never sees (11-21, README §7.3). The v3 request asks the teacher to write as the author of the
material, and a rewrite that still points at a hidden source is dropped. Both halves are tested here, and so is the
rule that a batch keeps the request wording it was made with.

    source scripts/env.sh && $MIDTRAIN_PYTHON -m pytest tests/test_synthetic_rephrase.py -q
"""
import pytest

from midtrain import synthetic_rephrase


def test_rewrites_that_point_at_a_hidden_source_are_dropped():
    source_text = "The ward had 1,087 residents in 2011."
    body = " ".join(["The ward counted its residents carefully and recorded the result in the census."] * 30)
    for reference in ("The text states that the ward had 1,087 residents.", "Student: The passage notes that it grew.",
                      "As described in the text, the count rose.", "Check: the text explicitly lists the wards."):
        assert synthetic_rephrase.reason_to_drop(f"{body} {reference}", source_text, "stop") == "refers_to_hidden_source"


def test_ordinary_uses_of_text_passage_and_author_are_kept():
    source_text = "The act passed in 2010."
    body = " ".join(["The reform changed how insurers priced their plans and who could buy them."] * 30)
    for sentence in ("It came with the passage of the Affordable Care Act in 2010.", "The text of the treaty was signed.",
                     "The author of the novel lived in Oslo.", "Features of the source domain are aligned."):
        assert synthetic_rephrase.reason_to_drop(f"{body} {sentence}", source_text, "stop") is None


def test_each_batch_keeps_the_request_wording_it_was_made_with():
    passage = {"passage": "Text of the passage.", "parent_doc_id": "d1", "parent_source": "wiki_en"}
    style_name = "structured_notes"
    first_batch = synthetic_rephrase.build_request_text(passage, style_name)
    second_batch = synthetic_rephrase.build_request_text(
        {**passage, "reader": "a teacher preparing a lesson", "writer_voice": "a technical writer"}, style_name)
    third_batch = synthetic_rephrase.build_request_text({**passage, "framing": "own_voice_specific"}, style_name)
    assert first_batch.startswith(synthetic_rephrase.style_instructions[style_name])
    assert second_batch.startswith("Write for a teacher preparing a lesson, in the voice of a technical writer.")
    assert third_batch.startswith(synthetic_rephrase.own_voice_instruction)
    assert all(request.endswith("Passage:\nText of the passage.") for request in (first_batch, second_batch, third_batch))


@pytest.mark.parametrize("answer_text, is_checkable", [
    ("42", True), ("1,250", True), ("\\frac{3}{4}", True), ("2\\sqrt{3}", True), ("-0.5", True),
    ("", False), ("3, 4", False), ("x=1 or x=2", False), ("1; 2", False), ("\\pm 2", False),
    ("The answer is the number of apples left over", False),
])
def test_only_single_short_answers_of_the_second_problem_set_are_kept(answer_text, is_checkable):
    """A list of values or two cases cannot be compared with one boxed answer, so such problems are not drawn."""
    assert synthetic_rephrase.numina_answer_is_checkable(answer_text) is is_checkable
