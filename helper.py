# contains all the helper functions for our project

def make_slot_dict(sent):
    slot_dict = dict()
    sent = sent.split('--')
    
    for s in sent:
        s = s.split('=')
        if len(s) > 1:
            slot_dict[s[0]] = s[1]
    
    return slot_dict
