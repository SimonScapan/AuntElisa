def switcher(string, FOO=''):
    from random import randint
    wort = {        
        'cant' : {
            1:"How do you know you can't " +  FOO + '?',
            2:'Perhaps you could' + FOO + ' if you tried.',
            3:'What would it take for you to' +  FOO + '?',
        },

        'am' : {
            1:'Did you come to me because you are ' + FOO + '?',
            2:'How long have you been ' + FOO + '?',
            3:'How do you feel about being ' + FOO + '?',
            4:'How does being FOO make you feel?',
            5:'Do you enjoy being ' + FOO + '?',
            6:"Why do you tell me you're " + FOO + '?',
            7:"Why do you think you're" + FOO + '?'
        },

        'sorry' : {
            1:'There are many times when no apology is needed.',
            2:'What feelings do you have when you apologize?'
        },

        'hello' : {
            1:"Hello... I'm glad you could drop by today.",
            2:'Hi there... how are you today?',
            3:'Hello, how are you feeling today?'
        },

        'think' : {
            1:'Do you doubt ' +  FOO + '?',
            2:'Do you really think so?',
            3:"But you're not sure " + FOO + '?'
        },

        'yes' : {
            1:'You seem quite sure.',
            2:'OK, but can you elaborate a bit?'
        },

        'my' : {
            1:'I see, your' + FOO + '.',
            2:'Why do you say that your ' + FOO + '?',
            3:'When your ' + FOO + ' , how do you feel?'
        },

        'you' : {
            1:'We should be discussing you, not me.',
            2:'Why do you say that about me?',
            3:'Why do you care whether I ' + FOO + '?'
        },

        'mother' : {
            1:'Tell me more about your mother.',
            2:'What was your relationship with your mother like?',
            3:'How do you feel about your mother?',
            4:'How does this relate to your feelings today?',
            5:'Good family relations are important.'
        },

        'quit' : {
            1:'Thank you for talking with me.',
            2:'Good-bye.',
            3:'Thank you, that will be $150.  Have a good day!'
        },

        'nothing' : {
            1:'Please tell me more.',
            2:"Let's change focus a bit... Tell me about your family.",
            3:'Can you elaborate on that?',
            4:'Why do you say that ' + FOO + '?',
            5:'I see.',
            6:'Very interesting.',
            7: FOO + '.',
            8:'I see.  And what does that tell you?',
            9:'How does that make you feel?',
            10:'How do you feel when you say that?'
        },
        
        'need' : {
            1:'Why do you need ' + FOO + '?',
            2:'Would it really help you to get ' + FOO + '?',
            3:'Are you sure you need ' + FOO + '?'
        },
        
        "whydontyou" : {
            1:"Do you really think I don't " + FOO + "?",
            2:'Perhaps eventually I will ' + FOO + '.',
            3:'Do you really want me to ' + FOO + '?'
        },
        
        'whycantI' : {
            1:'Do you think you should be able to ' + FOO + '?',
            2:'If you could FOO, what would you do?',
            3:"I don't know -- why can't you " + FOO + '?',
            4:'Have you really tried?'
        },
        
        'areyou' : {
            1:'Why does it matter whether I am ' + FOO + '?',
            2:'Would you prefer it if I were not ' + FOO + '?',
            3:'Perhaps you believe I am ' + FOO + '.',
            4:'I may be ' + FOO + ' -- what do you think?'
        },
        
        'what' : {
            1:'Why do you ask?',
            2:'How would an answer to that help you?',
            3:'What do you think?'
        },
        
        'how' : {
            1:'How do you suppose?',
            2:'Perhaps you can answer your own question.',
            3:"What is it you're really asking?"
        },
        
        'because' : {
            1:'Is that the real reason?',
            2:'What other reasons come to mind?',
            3:'Does that reason apply to anything else?',
            4:'If ' + FOO + ' , what else must be true?'
        },
        
        'friend' : {
            1:'Tell me more about your friends.',
            2:'When you think of a friend, what comes to mind?',
            3:"Why don't you tell me about a childhood friend?"
        },
        
        'computer' : {
            1:'Are you really talking about me?',
            2:'Does it seem strange to talk to a computer?',
            3:'How do computers make you feel?',
            4:'Do you feel threatened by computers?'
        },
        
        'isit' : {
            1:'Do you think it is ' + FOO + '?',
            2:"Perhaps it's " + FOO + ' -- what do you think?',
            3:'If it were ' + FOO + ' , what would you do?',
            4:'It could well be that ' + FOO + '.'
        },
        
        'itis' : {
            1:'You seem very certain.',
            2:"If I told you that it probably isn't " + FOO + " , what would you feel?"
        },
        
        'canyou' : {
            1:"What makes you think I can't " + FOO + '?',
            2:"If I could " + FOO + " , then what?",
            3:"Why do you ask if I can " + FOO + '?'
        },
        
        'cani' : {
            1:"Perhaps you don't want to " + FOO + '.',
            2:'Do you want to be able to ' + FOO + '?',
            3:'If you could ' + FOO + ' , would you?',
        },
        
        'youare' : {
            1:'Why do you think I am ' + FOO + '?',
            2:"Does it please you to think that I'm " + FOO + '?',
            3:'Perhaps you would like me to be ' + FOO + '.',
            4:"Perhaps you're really talking about yourself?",
            5:'Why do you say I am ' + FOO + '?',
            6:'Why do you think I am ' + FOO + '?',
            7:"Are we talking about you, or me?"
        },
        
        'Idont' : {
            1:"Don't you really FOO?",
            2:"Why don't you FOO?",
            3:"Do you want to FOO?"
        },
        
        'ifeel' : {
            1:"Good, tell me more about these feelings.",
            2:'Do you often feel ' + FOO + '?',
            3:'When do you usually feel ' + FOO + '?',
            4:'When you feel ' + FOO + ' , what do you do?'
        },
        
        'ihave' : {
            1:"Why do you tell me that you've " + FOO + '?',
            2:'Have you really ' + FOO + '?',
            3:'Now that you have ' + FOO + ' , what will you do next?'
        },
        
        'iwould' : {
            1:'Could you explain why you would ' + FOO + '?',
            2:'Why would you ' + FOO + '?',
            3:'Who else knows that you would ' + FOO + '?'
        },
        
        'isthere' : {
            1:'Do you think there is ' + FOO + '?',
            2:"It's likely that there is " + FOO + '.',
            3:'Would you like there to be ' + FOO + '?'
        },
        
        'why' : {
            1:"Why don't you tell me the reason why " + FOO + '?',
            2:'Why do you think ' + FOO + '?'
        },
        
        'iwant' : {
            1:'What would it mean to you if you got ' + FOO + '?',
            2:'Why do you want ' + FOO + '?',
            3:'What would you do if you got ' + FOO + '?',
            4:'If you got ' + FOO + ' , then what would you do?'
        },
        
        'father' : {
            1:'Tell me more about your father.',
            2:'How did your father make you feel?',
            3:'How do you feel about your father?',
            4:'Does your relationship with your father relate to your feelings today?',
            5:'Do you have trouble showing affection with your family?'
        },
        
        'child' : {
            1:'Did you have close friends as a child?',
            2:'What is your favorite childhood memory?',
            3:'Do you remember any dreams or nightmares from childhood?',
            4:'Did the other children sometimes tease you?',
            5:'How do you think your childhood experiences relate to your feelings today?'
        },
        
        '?' : {
            1:'Why do you ask that?',
            2:'Please consider whether you can answer your own question.',
            3:'Perhaps the answer lies within yourself?',
            4:"Why don't you tell me?"
        }       
    
    }
    zahl = randint(1, len(wort[string]))
    return print(wort[string][zahl])