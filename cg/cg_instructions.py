#!/usr/bin/env python3
"""cg_instructions.py -- natural-language instruction banks for the adjacency-grid experiment (Experiment A).

Mirrors the Section 26/27 design:
  * SEEN strings are used in training.
  * UNSEEN strings are never trained; they test language generalisation for trained operations.
    Each bank ends with strings that need a small inference step (the Section 26 failures were of
    this kind: "two right angles", "three quarter turns").
  * HELD-OUT operations / compositions have evaluation strings only.

PREAMBLE is a fixed description of the picture. Prepend it to every instruction in every arm
(DG arms and non-DG baselines) so the vocabulary "caller / callee / selected" is grounded; it is
identical for all items, so it cannot leak the answer.
"""
from typing import Dict, List

def preamble(n: int = 16) -> str:
    """The fixed description of the picture for an n-symbol grid (n = 16, 32 or 48)."""
    return (
        f"The image is a {n}x{n} call-graph matrix of {n} code symbols. A dark cell in row i, column j means "
        "symbol i calls symbol j. Diagonal cells show each symbol's kind: green function, teal method, "
        "purple class, orange test. The red diagonal cell is the selected symbol; yellow marks highlighted "
        "symbols; a blank diagonal cell means the symbol is absent."
    )


PREAMBLE = preamble(16)

SEEN: Dict[str, List[str]] = {
    "identity": [
        "Reproduce the call graph exactly as given.",
        "Copy the graph unchanged.",
        "Leave every symbol and every call as it is.",
        "Output the same matrix with no edits.",
        "Keep the graph exactly as shown.",
        "Make no change to the call graph.",
    ],
    "reverse_edges": [
        "Reverse the direction of every call edge.",
        "Flip each call so the callee calls the caller.",
        "Invert all edges of the call graph.",
        "Turn every call from A to B into a call from B to A.",
        "Transpose the call matrix.",
        "Point every edge the opposite way.",
    ],
    "reverse_order": [
        "List the symbols in reverse order, keeping the same calls.",
        "Renumber the symbols from last to first.",
        "Reverse the ordering of the symbols without changing the graph.",
        "Relabel the nodes so the final symbol comes first and the first comes last.",
        "Put the symbols in the opposite order; calls stay between the same symbols.",
        "Reorder the rows and columns back to front.",
    ],
    "delete_selected": [
        "Delete the selected symbol and all of its calls.",
        "Remove the selected symbol from the graph.",
        "Drop the red symbol together with every edge that touches it.",
        "Erase the selected node and its incoming and outgoing calls.",
        "Take the selected symbol out of the codebase.",
        "Remove the highlighted red node and its edges.",
    ],
    "keep_callees": [
        "Keep only the selected symbol and the symbols it calls directly; remove the rest.",
        "Extract the selected symbol together with its direct callees.",
        "Reduce the graph to the red symbol and what it calls.",
        "Remove every symbol that the selected one does not call directly.",
        "Keep the selected symbol and its immediate callees only.",
        "Show just the selected symbol and the functions it invokes itself.",
    ],
    "mark_reachable": [
        "Mark every symbol the selected one can reach through one or more calls.",
        "Highlight all symbols reachable from the selected symbol.",
        "Mark the selected symbol's direct and indirect callees.",
        "Highlight everything downstream of the red symbol in the call graph.",
        "Mark all descendants of the selected symbol.",
        "Highlight each symbol that the selected one calls, directly or through other symbols.",
    ],
    "delete_tests": [
        "Delete all test symbols and their calls.",
        "Remove every test from the graph.",
        "Drop the orange symbols together with their edges.",
        "Strip the test code out of the call graph.",
        "Erase each test symbol and every call it makes or receives.",
        "Keep only non-test symbols.",
    ],
    "delete_classes": [
        "Delete all class symbols and their calls.",
        "Remove every class from the graph.",
        "Drop the purple symbols together with their edges.",
        "Strip the classes out of the call graph.",
        "Erase each class symbol and every call it makes or receives.",
        "Keep only symbols that are not classes.",
    ],
    "selected_first": [
        "Move the selected symbol to the first position; keep the others in their current order.",
        "Reorder the symbols so the selected one comes first.",
        "Bring the red symbol to the front of the ordering.",
        "Renumber the symbols with the selected one as number one and the rest unchanged in order.",
        "Put the selected symbol at the top-left and shift the others down.",
        "Make the selected symbol the first row and first column.",
    ],
}

UNSEEN: Dict[str, List[str]] = {
    "identity": [
        "Return the input graph as it stands.",
        "Do nothing to the matrix.",
        "Give back an untouched copy of the call graph.",
        "Apply the edge reversal twice.",                                   # inference
    ],
    "reverse_edges": [
        "Swap the roles of caller and callee in every call.",
        "Make each edge run backwards.",
        "Mirror the matrix across its main diagonal.",
        "Whatever a symbol calls should now call it instead, and nothing else.",   # inference
    ],
    "reverse_order": [
        "Number the symbols backwards.",
        "Flip the sequence of the symbols end to end.",
        "Rotate the matrix by half a turn.",                                 # inference: rot180 == reverse order
        "The sixteenth symbol becomes the first, the fifteenth the second, and so on.",
    ],
    "delete_selected": [
        "Get rid of the selected symbol entirely.",
        "The selected function is dead code; take it out.",
        "Clear the row and the column of the red symbol, including its diagonal cell.",
        "Leave fifteen symbols by removing the one that is selected.",       # inference
    ],
    "keep_callees": [
        "Narrow the graph down to the selected symbol plus whatever it calls itself.",
        "Discard all symbols except the selected one and its one-hop callees.",
        "Keep the red symbol and the symbols whose columns are dark in its row.",
        "Retain only what the selected symbol depends on directly, and the symbol itself.",   # inference
    ],
    "mark_reachable": [
        "Highlight the transitive callees of the selected symbol.",
        "Mark whatever could end up running when the selected symbol runs.",
        "Flag every symbol that lies on a call path starting at the red symbol.",
        "Highlight the callees of the selected symbol, their callees, and so on until nothing new appears.",  # inference
    ],
    "delete_tests": [
        "Take out the tests.",
        "Remove all symbols of the test kind.",
        "Clear every row and column whose diagonal cell is orange.",
        "Leave only library code: functions, methods and classes.",          # inference
    ],
    "delete_classes": [
        "Take out the classes.",
        "Remove all symbols of the class kind.",
        "Clear every row and column whose diagonal cell is purple.",
        "Leave only functions, methods and tests.",                          # inference
    ],
    "selected_first": [
        "Shift the selected symbol to position one.",
        "Lead the ordering with the selected symbol.",
        "Swap nothing else, but move the red symbol's row and column to the start.",
        "The selected symbol should come before every other symbol; do not reorder the others.",   # inference
    ],
}

# The unseen strings that need an inference step, one per trained operation (analogue of the Section 27
# "arithmetic" paraphrases such as "two right angles"). Each is also the last entry of its UNSEEN bank.
INFERENCE_UNSEEN: Dict[str, str] = {op: bank[-1] for op, bank in UNSEEN.items()}

HELDOUT_OP_STRINGS: Dict[str, List[str]] = {
    "keep_callers": [
        "Keep only the selected symbol and the symbols that call it directly; remove the rest.",
        "Extract the selected symbol together with its direct callers.",
        "Reduce the graph to the red symbol and what calls it.",
        "Remove every symbol that does not call the selected one directly.",
    ],
    "delete_methods": [
        "Delete all method symbols and their calls.",
        "Remove every method from the graph.",
        "Drop the teal symbols together with their edges.",
        "Strip the methods out of the call graph.",
    ],
    "mark_ancestors": [
        "Mark every symbol that can reach the selected one through one or more calls.",
        "Highlight all symbols from which the selected symbol is reachable.",
        "Mark the selected symbol's direct and indirect callers.",
        "Highlight everything upstream of the red symbol in the call graph.",
    ],
}

# Two-step compositions: key "first+second".
COMP_SEEN: Dict[str, List[str]] = {
    "reverse_edges+reverse_order": [
        "Reverse the direction of every call edge, then list the symbols in reverse order.",
        "First invert all edges, then renumber the symbols from last to first.",
        "Flip each call, and after that reverse the ordering of the symbols.",
        "Transpose the call matrix and then reorder the rows and columns back to front.",
    ],
    "reverse_edges+delete_selected": [
        "Reverse the direction of every call edge, then delete the selected symbol and all of its calls.",
        "First invert all edges, then remove the selected symbol from the graph.",
        "Flip each call, and after that drop the red symbol with every edge that touches it.",
        "Transpose the call matrix and then erase the selected node.",
    ],
    "reverse_order+keep_callees": [
        "List the symbols in reverse order, then keep only the selected symbol and the symbols it calls directly.",
        "First renumber the symbols from last to first, then extract the selected symbol with its direct callees.",
        "Reverse the ordering of the symbols, and after that reduce the graph to the red symbol and what it calls.",
        "Reorder the rows and columns back to front and then keep the selected symbol and its immediate callees only.",
    ],
    "reverse_order+mark_reachable": [
        "List the symbols in reverse order, then mark every symbol the selected one can reach.",
        "First renumber the symbols from last to first, then highlight all symbols reachable from the selected symbol.",
        "Reverse the ordering of the symbols, and after that mark all descendants of the selected symbol.",
        "Reorder the rows and columns back to front and then highlight everything downstream of the red symbol.",
    ],
    "reverse_edges+delete_tests": [
        "Reverse the direction of every call edge, then delete all test symbols and their calls.",
        "First invert all edges, then remove every test from the graph.",
        "Flip each call, and after that drop the orange symbols together with their edges.",
        "Transpose the call matrix and then strip out the test code.",
    ],
}
COMP_UNSEEN: Dict[str, List[str]] = {
    "reverse_edges+reverse_order": [
        "Make each edge run backwards and afterwards number the symbols backwards.",
        "Swap caller and callee everywhere; once that is done, flip the sequence of the symbols end to end.",
    ],
    "reverse_edges+delete_selected": [
        "Make each edge run backwards and afterwards get rid of the selected symbol entirely.",
        "Swap caller and callee everywhere; once that is done, take the red symbol out.",
    ],
    "reverse_order+keep_callees": [
        "Number the symbols backwards and afterwards discard everything except the selected symbol and its one-hop callees.",
        "Flip the sequence of the symbols end to end; once that is done, narrow the graph to the selected symbol plus what it calls itself.",
    ],
    "reverse_order+mark_reachable": [
        "Number the symbols backwards and afterwards highlight the transitive callees of the selected symbol.",
        "Flip the sequence of the symbols end to end; once that is done, flag every symbol on a call path starting at the red symbol.",
    ],
    "reverse_edges+delete_tests": [
        "Make each edge run backwards and afterwards take out the tests.",
        "Swap caller and callee everywhere; once that is done, remove all symbols of the test kind.",
    ],
}
HELDOUT_COMP_STRINGS: Dict[str, List[str]] = {
    "reverse_order+delete_selected": [
        "List the symbols in reverse order, then delete the selected symbol and all of its calls.",
        "First renumber the symbols from last to first, then remove the selected symbol from the graph.",
        "Reverse the ordering of the symbols, and after that drop the red symbol with every edge that touches it.",
        "Reorder the rows and columns back to front and then erase the selected node.",
    ],
    "reverse_edges+keep_callees": [
        "Reverse the direction of every call edge, then keep only the selected symbol and the symbols it calls directly.",
        "First invert all edges, then extract the selected symbol with its direct callees.",
        "Flip each call, and after that reduce the graph to the red symbol and what it calls.",
        "Transpose the call matrix and then keep the selected symbol and its immediate callees only.",
    ],
    "reverse_edges+mark_reachable": [
        "Reverse the direction of every call edge, then mark every symbol the selected one can reach.",
        "First invert all edges, then highlight all symbols reachable from the selected symbol.",
        "Flip each call, and after that mark all descendants of the selected symbol.",
        "Transpose the call matrix and then highlight everything downstream of the red symbol.",
    ],
    "reverse_order+delete_tests": [
        "List the symbols in reverse order, then delete all test symbols and their calls.",
        "First renumber the symbols from last to first, then remove every test from the graph.",
        "Reverse the ordering of the symbols, and after that drop the orange symbols together with their edges.",
        "Reorder the rows and columns back to front and then strip out the test code.",
    ],
    "delete_tests+mark_reachable": [
        "Delete all test symbols and their calls, then mark every symbol the selected one can reach.",
        "First remove every test from the graph, then highlight all symbols reachable from the selected symbol.",
        "Drop the orange symbols with their edges, and after that mark all descendants of the selected symbol.",
        "Strip out the test code and then highlight everything downstream of the red symbol.",
    ],
}

# Open-ended operations: several outputs are valid (HANDOFF section 12.1). Not part of all_strings().
OPEN_SEEN: Dict[str, List[str]] = {
    "delete_one_test": [
        "Delete exactly one test symbol; any one will do.",
        "Remove a single test from the graph and leave the other tests.",
        "Drop one of the orange symbols, together with its edges.",
        "Pick any one test symbol and delete it.",
        "Erase one test, whichever you like, and nothing else.",
        "Take out just one of the tests.",
    ],
    "delete_one_callee": [
        "Delete exactly one of the symbols the selected symbol calls; any one will do.",
        "Remove a single direct callee of the selected symbol.",
        "Drop one of the red symbol's callees, together with its edges.",
        "Pick any one symbol that the selected symbol calls and delete it.",
        "Erase one callee of the selected symbol, whichever you like, and nothing else.",
        "Take out just one of the symbols called by the selected one.",
    ],
    "cut_one_call": [
        "Remove exactly one call made by the selected symbol; any one will do. Keep both symbols.",
        "Cut a single outgoing edge of the selected symbol.",
        "Drop one of the calls the red symbol makes, without deleting any symbol.",
        "Pick any one call from the selected symbol and remove that edge.",
        "Erase one outgoing call of the selected symbol, whichever you like, and nothing else.",
        "Take out just one of the edges that start at the selected symbol.",
    ],
}
OPEN_UNSEEN: Dict[str, List[str]] = {
    "delete_one_test": [
        "Get rid of one test; it does not matter which.",
        "There should be one test fewer afterwards.",
        "Choose a test symbol and clear its row, column and diagonal cell.",
        "Reduce the number of orange symbols by one.",
    ],
    "delete_one_callee": [
        "Get rid of one of the selected symbol's callees; it does not matter which.",
        "The selected symbol should end up with one callee fewer because that symbol is gone.",
        "Choose a dark cell in the red symbol's row and delete the symbol in that column.",
        "Reduce the number of symbols the selected one calls by deleting one of them.",
    ],
    "cut_one_call": [
        "Get rid of one of the selected symbol's outgoing calls; it does not matter which.",
        "The selected symbol should make one call fewer, and no symbol should disappear.",
        "Choose a dark cell in the red symbol's row and clear only that cell.",
        "Reduce the number of calls the selected symbol makes by one.",
    ],
}


def open_strings() -> List[str]:
    return [x for d in (OPEN_SEEN, OPEN_UNSEEN) for v in d.values() for x in v]


# For each condition, also ask DG the condition as a TEXT question on the same image (Section 27 Part 3).
# Answers are computed by cg_grid_dataset.text_question_answers().
TEXT_QUESTIONS: Dict[str, str] = {
    "n_callees": "How many symbols does the selected symbol call directly? Answer with a number.",
    "n_callers": "How many symbols call the selected symbol directly? Answer with a number.",
    "n_reachable": "How many symbols can the selected symbol reach through one or more calls? Answer with a number.",
    "n_tests": "How many test symbols (orange diagonal cells) are in the graph? Answer with a number.",
    "selected_position": "Counting from 1 at the top-left, which position on the diagonal is the selected symbol? Answer with a number.",
}


def bank_for(target: str, split: str) -> List[str]:
    """split in {"seen", "unseen", "heldout"}."""
    if split == "seen":
        return SEEN.get(target) or COMP_SEEN[target]
    if split == "unseen":
        return UNSEEN.get(target) or COMP_UNSEEN[target]
    return HELDOUT_OP_STRINGS.get(target) or HELDOUT_COMP_STRINGS[target]


def all_strings() -> List[str]:
    out = []
    for d in (SEEN, UNSEEN, HELDOUT_OP_STRINGS, COMP_SEEN, COMP_UNSEEN, HELDOUT_COMP_STRINGS):
        for v in d.values():
            out.extend(v)
    return out
