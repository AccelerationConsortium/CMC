def make_OTFlex_protocol(list_of_exps, template_file_path, output_file_path, solvent_mix_loc, row):
    import re

    # Read the original protocol
    with open(template_file_path, 'r') as f:
        protocol_code = f.read()

    # Replace the exp_list block
    new_exp_list_block = f"exp_list = {repr(list_of_exps)}"
    protocol_code = re.sub(
        r"exp_list\s*=\s*{.*?}\n",
        new_exp_list_block + "\n",
        protocol_code,
        flags=re.DOTALL
    )

    # Replace solvent_mix_loc
    protocol_code = re.sub(
        r"solvent_mix_loc\s*=\s*['\"].*?['\"]",
        f"solvent_mix_loc = '{solvent_mix_loc}'",
        protocol_code
    )

    # Replace row
    protocol_code = re.sub(
        r"row\s*=\s*['\"].*?['\"]",
        f"row = '{row}'",
        protocol_code
    )

    # Save to a new file
    with open(output_file_path, 'w') as f:
        f.write(protocol_code)
