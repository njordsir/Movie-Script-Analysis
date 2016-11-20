# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 22:18:20 2016

@author: naman
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 20:36:16 2016

@author: naman
"""
import json
import re
from collections import Counter

def script_by_scenes(path_to_script):
    
    fp=open(path_to_script, 'r')
    script=fp.readlines()
    script=[re.sub("[\(\[].*?[\)\]]", "", re.sub('<.*?>', '', s).rstrip('\n').rstrip('\r').strip()).rstrip('\n').rstrip('\r').strip() for s in script]
    scene_break=list()
    scene_break=[l for l in script if l.isupper()]
    
    list_of_chars=dict(Counter(scene_break))
    list_of_chars=[k for k,v in list_of_chars.items() if (v>=5 and (k.find('CUT') == -1 and k.find('INT.') == -1 and k.find('EXT.') == -1 ))]
    scenes = dict()
    cntr=1
    scenes[1]=list()
    for l in scene_break:
        if l in list_of_chars:
            scenes[cntr].append(l)
        else:
            if len(scenes[cntr])!=0:
                cntr += 1
                scenes[cntr]=list()
    if len(scenes[cntr])==0:
        del scenes[cntr]
            
    return scenes
   
    


import os
scripts=os.listdir('/home/naman/SNLP/imsdb')

try: 
    os.makedirs('/home/naman/SNLP/database_new')
except OSError:
    if not os.path.isdir('/home/naman/SNLP/database_new'):
        raise

for cntr,script in enumerate(scripts):
    print "Parsing Script %s : %d/%d" % (script.strip('.txt'), cntr+1, len(scripts))
    with open('/home/naman/SNLP/database_new/%s.json' % script.strip('.txt'), 'w') as fp:        
        json.dump(script_by_scenes('/home/naman/SNLP/imsdb/%s'% script), fp)
    
