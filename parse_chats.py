"""
Reads the chat logs, removes the reactions, collects the questions and groups their responses.

TODO: Fix the ellipses bug for inline comments
"""
import argparse
import json
import pathlib
import re

def parse_reference(string):
    """
    Replying to "<question-snippet>..."<EOS><stuff>
    NOTE: Hard-coded to English
    """
    regex_double_quotes_ellipses = r"\"(.*?)\.\.\."
    match = re.search(regex_double_quotes_ellipses, string)
    if match:
        # .group(1) extracts the content inside the parentheses
        out = match.group(1).strip()
        return out

    return None

def read_file(fname):
    """
    Basic read lines (skips empty ones)
    """
    list_out = []
    with open(fname) as file:
        for line in file:
            line = line.strip()
            # skips
            if not line:
                continue
            list_out.append(line)
    
    return list_out

def extract_responses(list_lines, token_eos):
    """
    Each line is <HH:mm:ss>  From <name> <description> : <text>
    """
    list_out = []
    for line in list_lines:
        # parse
        if "	 From " in line:
            text = line.split(":", 3)[3].strip()
            list_out.append(text)
        else:
            list_out[-1] += token_eos + line
    
    return list_out

parser = argparse.ArgumentParser(description="zoom logs Q&A parser")
parser.add_argument("--file", type=str, required=True, help="ZOOM log to parse")
parser.add_argument("--out-file", type=str, required=False, help="Filename to save")
args = parser.parse_args()

fname = args.file

# fail fast
if not pathlib.Path(fname).resolve().exists:
    raise FileNotFoundError(fname)

list_reactions = ["Reacted to", "Se ha reaccionado a"]
token_eos = "<EOS>"
list_lines = read_file(fname)
list_lines = extract_responses(list_lines, token_eos)

# categorize lines
set_questions = set()
dict_questions = {}

token_eos = "<EOS>"
for line in list_lines:
    # skip
    if line.startswith(tuple(list_reactions)):
        continue

    # check if question
    if line.startswith("Replying to"): # is answer
        list_responses = line.split(token_eos)
        question = parse_reference(list_responses[0])
        if question is not None:
            snippet_responses = f"{token_eos}".join(list_responses[1:])
            if question in dict_questions:
                dict_questions[question].append(snippet_responses)
            else:
                dict_questions[question] = [snippet_responses]

    else: # is question
        set_questions.add(line)

# collect questions and responses
list_question_sizes = list(set(len(question) for question in dict_questions.keys()))
list_question_unanswered = []
for question in set_questions:
    found_key = None
    for question_size in list_question_sizes: # at most O(len(list_question_sizes)) lookups
        if question[:question_size] in dict_questions:
            found_key = question[:question_size]
            break # inner for-loop
    if found_key:
        dict_questions[question] = dict_questions.pop(found_key)
    else:
        # list_question_unanswered.append(question)
        dict_questions[question] = []

fname_out = args.out_file
if fname_out is None:
    fname_out = "chat_dump.json"
with open(fname_out, "w") as file_out:
    json.dump(dict_questions, file_out, indent=4)
# print(list_question_unanswered)