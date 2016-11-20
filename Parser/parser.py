# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 20:36:16 2016

@author: naman
"""
import json
import re


def script_by_scenes(path_to_script):
    
    fp=open(path_to_script, 'r')
    script=fp.readlines()
    
    script=[re.sub('<.*?>', '', s).rstrip('\n').rstrip('\r').strip() for s in script]
    
    
    def extra_conditions(string):
        
        if( (string.find('Cont') or string.find('cont') or string.find('CONTINUED') or string.find("CONT'D")) != -1):
            flag=0
        elif(string[0] == '(' != -1):
            flag=0
        else:
            flag=1
        return flag
        
    
    
    
    portions=list()
    portion_id=0
    portion_text=list()
    
    
    
    for line_id, line in enumerate(script):
        portion_text.append(line)
        
        if(line_id == len(script)-1):
            portions.append((portion_id, portion_text))
            portion_text=list()
            portion_id += 1
            break
        
        tmp_line=''.join(e for e in script[line_id+1] if e.isalpha())

        if(tmp_line.isupper() and extra_conditions(script[line_id+1])):
            portions.append((portion_id, portion_text))
            portion_text=list()
            portion_id += 1

    portions=[p for p in portions if p[1][0]!='']
    
    characters_tmp=[tmp[1][0] for tmp in portions]
    characters_tmp=[re.sub('\(.*?\)', '', tmp).strip() for tmp in characters_tmp]
    from collections import Counter
    characters_tmp=zip(Counter(characters_tmp).keys(), Counter(characters_tmp).values())
    characters=[tmp[0] for tmp in characters_tmp if tmp[1] > 10 and (tmp[0].find('CUT') == -1)]
    scene_names=[tmp[0] for tmp in characters_tmp if tmp[0] not in characters]
    
    
    scenes=list()
    scene_id=1
    scene=list()
    
    for portion in portions:
        scene.append(portion[1])
        if(portion[0] != len(portions)):
            next_name = re.sub('\(.*?\)', '', portions[portion[0]][1][0])
        else:
            break
        if next_name in scene_names:
            scenes.append((scene_id, scene))
            scene_id += 1
            scene=list()
            
    return scenes, characters, scene_names
   
    
def parse_data(path_to_script):  
  
    scenes, list_of_chars, list_of_scenes =script_by_scenes(path_to_script)
    all_scenes=dict()
    cntr=0
    
    for scene_content in scenes:
    
        scene = dict()
        scene_formatted = dict()
        scene_desc_list = list()
        char_dialogues = list()
        
        scene_part_id=1
      
        for scene_part in scene_content[1]:        
            
            if scene_part_id == 1:
                scene_part_type = 'SCENE_DESC'
                scene_part_name = scene_part[0]
                scene_part_content = ""
                for sp in scene_part[1:]:
                    scene_part_content += sp + " " 
                scene[scene_part_id] = {scene_part_name : (scene_part_type, scene_part_content)} 
                scene_part_id += 1
                continue    
            
            if re.sub('\(.*?\)', '', scene_part[0]).strip() in list_of_chars:
                scene_part_type = 'DIALOGUE'
                
            else:
                #print "Scene_Part_Type should only be dialogues of chars: %s" % scene_part[0]
                break
                
            scene_part_name = re.sub('\(.*?\)', '', scene_part[0]).strip()
            dialogue = ""       
            
            flag=1
            index=-1
            for idx,sp in enumerate(scene_part[1:]):
                
                if sp == '':
                    index = idx +1
                    break
                if sp[0] == '(' and sp.find(')') != -1:
                    flag = 1
                    dialogue += sp + '\n'
                    continue
                elif sp[0] == '(' and sp.find(')') == -1:
                    flag = 0
                    dialogue += sp + ' '
                    continue
                if flag == 0 and sp.find(')') != -1:
                    flag = 1
                    dialogue += sp + '\n'
                    continue
                if flag == 0 and sp.find(')') == -1:
                    flag = 0
                    dialogue += sp + ' '
                    continue
                
                dialogue += sp + ' '
            
            scene[scene_part_id] = {scene_part_name : (scene_part_type, dialogue)} 
            scene_part_id += 1
                   
                   
            if index+1 >= len(scene_part):
                break
    
            desc_following_dialogue = "" 
            scene_part_type = "SCENE_DESC_FOLLOWING_DIALOGUE"
            scene_part_name = re.sub('\(.*?\)', '', scene_part[0]).strip()
    
            for idx,sp in enumerate(scene_part[index+1:]):
                desc_following_dialogue += sp + " "
            
            scene[scene_part_id] = {scene_part_name : (scene_part_type, desc_following_dialogue)} 
            scene_part_id += 1
            
            
    
        
        for k,v in scene.items():
            if v.values()[0][0] == 'SCENE_DESC_FOLLOWING_DIALOGUE' or v.values()[0][0] == 'SCENE_DESC':
                scene_desc_list.append([k, v.values()[0][1]])
            elif v.values()[0][0] == 'DIALOGUE':
                char_dialogues.append([k, v.keys()[0], v.values()[0][1]])
    
        scene_formatted['scene_descriptons_list'] = scene_desc_list
        scene_formatted['char_dialogues'] = char_dialogues
        if len(char_dialogues) > 0:
            cntr += 1
        all_scenes[scene_content[0]] = scene_formatted
    
    print "All Scenes Processed"
    
    if len(all_scenes) == 0:
        return all_scenes, -1
        
    
    print cntr*1.0/len(all_scenes)
    
    return all_scenes, cntr*1.0/len(all_scenes)

import os
scripts=os.listdir('/home/naman/SNLP/imsdb')

try: 
    os.makedirs('/home/naman/SNLP/database_new')
except OSError:
    if not os.path.isdir('/home/naman/SNLP/database_new'):
        raise

for cntr,script in enumerate(scripts):
    print "Parsing Script %s : %d/%d" % (script.strip('.txt'), cntr+1, len(scripts))
    processed_scenes, flag = parse_data('/home/naman/SNLP/imsdb/%s' % script)
    if flag >= 0.75:
        with open('/home/naman/SNLP/database_new/%s.json' % script.strip('.txt'), 'w') as fp:        
            json.dump(processed_scenes, fp)
    
