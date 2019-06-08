INTRO = (
    '\n'
    'Analyzing Message Popularity in Group Chats\n'
    '-- Ritik Shah and Winston Chen\n'
    '\n'
    "This program analyzes messages from a groupchat/group settings.\n"
    "There are two datasets available:\n"
    ' - freecodecamp : primary\n'
    ' - groupme : secondary\n'
    '\n'
    'Which dataset would you like to analyze?\n'
)

GROUPME = (
    "The Groupme dataset is a private dataset from our groupchat messages.\n"
    "The original data included user information, text, and the likes for "
    "each message sent.\n"
    "The data is anonymized and preprocessed to protect the information.\n"
    "This dataset is also quite small so only expect it to take 3 to 4 min\n"
    '\n'
    'Press enter to start the machine learning process on the groupme data\n'
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
    '4. Run from a serialized pickled file\n'
    '\n'
    'These last two require serialized pickles or it will fail\n'
    '(running 1 through 3 produces the pickles at those data levels)\n'
    '5. Run the machine learning test file for hyperparameters\n'
    '6. Run the feature analysis suite.\n'
    '\n'
    'Typing anything else will quit the program\n'
)
