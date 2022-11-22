File Contents are:

SBC001.trn --> Orginal Audio file with all the tags
output.txt --> output after cleaning and tagging
disfluency_key.json --> Json file of all keys/vals that we decide for tagging. (we only choose a subset of them)

So the corresponding TAGS I chose for the sample output from one file in the Santa Barbara Corpus are:

(Information can be found in disfluency_key.json) 

"Pause": {
        "...(N)": "(long)",
        "...": "(medium)",
        "..": "(short)",
        "(0)": "(latching)"
    },
"Vocal Noises": {
        "( )": "(vocal_noises)",
        "(H)": "(inhalation)",
        "(Hx)": "(exhalation)",
        "%": "(glottal_stop)",
        "@": "(laughter)"
    },

"Extra": {
    	"um": "(um)",
    	"umm": "(umm)",
    	"uh": "(uh)",
    	"uhh": "(uhh)"

What my code did is take the keys, finds them in the text using regex, then replaces them with the values on the left. (Then cleans up most everything else) 

So those are the keys you should look out for (for now). Anything else in parenthesis that you attempt to capture would most likely be garbage that slipped through. 


