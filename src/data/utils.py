from typing import List


def rows_to_chat(rows: List[List[dict]]) -> List[dict]:
    """
    Converts a list of rows to a list of chats. Each row is a list of messages.

    We assume that the following columns exist in the rows:
    - 'system_prompt'
    - 'one_shot_user_input'
    - 'one_shot_assistant_output'
    - 'user_input'
    """

    chats = []
    for system_prompt, one_shot_user_input, one_shot_assistant_output, user_input in \
        zip(rows['system_prompt'], rows['one_shot_user_input'], rows['one_shot_assistant_output'], rows['user_input']):
        chats.append([
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': one_shot_user_input},
            {'role': 'assistant', 'content': one_shot_assistant_output},
            {'role': 'user', 'content': user_input}
        ])

    return chats
