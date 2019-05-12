# =============================================================================
# CSE 163 Final Project
# ~ Ritik Shah and Winston Chen ~
# =============================================================================

from internal.process import process


def main(in_name, out_name):
    process(in_name, out_name)


if __name__ == "__main__":
    in_file = 'data/freecodecamp_casual_chatroom.csv'
    out_file = 'data/data.json'
    main(in_file, out_file)
