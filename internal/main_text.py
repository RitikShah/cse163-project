INTRO = (
    '\n'
    'Analyzing Message Popularity in Group Chats\n'
    '-- Ritik Shah and Winston Chen\n'
    '\n'
    "This program analyzes messages from a groupchat setting.\n"
    "There are two datasets available:\n"
    ' - freecodecamp : primary dataset (do this first)\n'
    ' - groupme\n'
    '\n'
    'Which dataset would you like to analyze?\n'
)

GROUPME = (
    "The Groupme dataset is a private dataset from our groupchat messages.\n"
    "The original data included user information, text, and the likes for "
    "each message sent.\n"
    "The data is anonymized and preprocessed to protect the information.\n"
    '\n'
    'Press enter to start the machine learning process on groupme (1-2 min)\n'
    'Typing anything else will quit the program\n'
)

FREECODECAMP = (
    'The FreeCodeCamp dataset is a public dataset uploaded on Kaggle\n'
    "Due to the size of the dataset, running the whole set is inadvisable\n"
    "However, we provide various options of running the program.\n"
    '\n'
    'Type the number associated with the action\n'
    "1. 100% of the data | 3-4 hours and up to 50gb of memory (yea...)\n"
    '2. 10%  of the data | 10-15 minutes and up to 6gb of memory\n'
    '3. 1%   of the data | 1-2 minutes and up to 900mb memory\n'
    '4. Run from pickles' # noqa
    '\n'
    'Typing anything else will quit the program\n'
)


test = (
    '1. Run everything from scratch\n'
    "    This runs the entire program from scratch without reading pkls\n"
    '    Warning, can take a sigificant amount of time.\n'
    '2. Read from cleaned.pkl\n'
    '    cleaned.pkl is a serialized file that contains the csv data cleaned\n'
    '     into a pandas DataFrame.\n'
    '3. Read from features.pkl\n'
    '    features.pkl is a serialized file that contains the data with the\n'
    '     text processed features as columns.\n'
    '4. Run the machine learning algorithm\n'
    '    Reads in the pickle files and produces the accuracy score rating\n'
    '     the model.\n'
    '5. Run the machine learning testing suite.\n'
    "    This file was ran independently to determine the model's\n"
    '     hyperparameters.\n'
    '\n'
    'The latter numbers will run best if you have already generated the\n'
    ' pickled files and wish to rerun a certain part of the program.\n'
    'When a number is chosen, it will continue to roll through the rest it\n'
    ' without user input.\n'
    '\n'
)
INTRO_INPUT = '> '
