def Therapist(eingabe):
    from switcher import switcher
    from random import randint
    import re
    
    eingabe = str(eingabe)
    
    eingabe = eingabe.replace("?","")
    eingabe = eingabe.replace("!","")
    eingabe = eingabe.replace(".","")
    eingabe = eingabe.lower()

    
    if re.search("i can't",eingabe):
        switcher("cant",eingabe[8:])
    
    elif re.search("i am",eingabe):
        switcher("am",eingabe[5:])
        
    elif re.search("sorry",eingabe):
        switcher("sorry")
        
    elif re.search("hello",eingabe):
        switcher("hello")
        
    elif re.search("i think",eingabe):
        switcher("think", eingabe[8:])

    elif re.search("yes",eingabe):
        switcher("yes")
        
    elif re.search("mother",eingabe):
        switcher("mother")
        
    elif re.search("my",eingabe):
        switcher("my", eingabe[3:])

    elif re.search("you",eingabe):
        switcher("you", eingabe[4:])
        
    elif re.search("quit",eingabe):
        switcher("quit")
    
    elif re.search("i need",eingabe):
        switcher("need", eingabe[7:])
    
    elif re.search("why don",eingabe):
        switcher("whydontyou", eingabe[10:])
    
    elif re.search("why can",eingabe):
        switcher("WhycantI", eingabe[9:])
    
    elif re.search("i'm",eingabe):
        switcher("am", eingabe[4:])
    
    elif re.search("are you",eingabe):
        switcher("areyou", eingabe[8:])
    
    elif re.search("what",eingabe):
        switcher("what")
        
    elif re.search("how",eingabe):
        switcher("how")
        
    elif re.search("because",eingabe):
        switcher("because")
        
    elif re.search("sorry",eingabe):
        switcher("sorry")
        
    elif re.search("hello",eingabe):
        switcher("hello")

    elif re.search("i think",eingabe):
        switcher("hello", eingabe[8:])
        
    elif re.search("friend",eingabe):
        switcher("friend")
                
    elif re.search("yes",eingabe):
        switcher("yes")

    elif re.search("computer",eingabe):
        switcher("computer")
                
    elif re.search("is it",eingabe):
        switcher("isit",eingabe[6:])
                
    elif re.search("it is",eingabe):
        switcher("itis",eingabe[6:])
                
    elif re.search("can you",eingabe):
        switcher("canyou", eingabe[8])
                
    elif re.search("can i",eingabe):
        switcher("cani", eingabe[6:])
            
    elif re.search("you are",eingabe):
        switcher("youare",eingabe[8:])

    elif re.search("you're",eingabe):
        switcher("youare",eingabe[7:])
                
    elif re.search("i don't",eingabe):
        switcher("Idont",eingabe[8:])
                
    elif re.search("i feel",eingabe):
        switcher("ifeel",eingabe[7:])
                
    elif re.search("i have",eingabe):
        switcher("ihave",eingabe[7:])
                
    elif re.search("i have",eingabe):
        switcher("ihave",eingabe[7:])
                
    elif re.search("i have",eingabe):
        switcher("ihave",eingabe[7:])
                
    elif re.search("i have",eingabe):
        switcher("ihave",eingabe[7:])
                
    elif re.search("i have",eingabe):
        switcher("ihave",eingabe[7:])
        
    else:
        switcher("nothing",eingabe)
print()