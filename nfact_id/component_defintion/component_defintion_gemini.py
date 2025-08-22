import os
from google import genai
import decouple
import json
import time
from tqdm import tqdm

def get_locations(path) -> dict:
    """
    Load json of locations

    Parameters
    -----------
    path: str
        path to dictionary
    
    Return
    -------
    dict: dictionary object
        dict of loactions by
        components
    """
    with open(path, "r") as f:
        return json.load(f)


def prompt() -> str:
    """
    Prompt to send to gemini

    Parameters
    -----------
    None

    Returns
    -------
    str: str 
        prompy string
    """
    
    return """
   You are an expert in neuroanatomy. I will give you a set of grey matter and white matter regions from analysis I have done. I want you to use your expertise to tell me what functional brain network this list of regionsbelongs to. The possbile list of functional networks is: 'Visual', 'Default', 'Dorsal Attention', 'Frontoparietal', 'Limbic', 'Somatomotor', 'Salience', 'Temporal Parietal','Language', 'Auditory', 'Posterior Multimodal', 'Ventral Multimodal'. 
   Not all regions will be relevant to the network so you must decide based on your expertise which regions are relevant and which ones aren"t. To help I have given the name of the region with a dice score.
   Provide your response in json format with the first key as network which is just the name of the network. The next key is a reson key where you will provide a very
   short summary of your reasoning. Finally I want a probability key were you will give a probability of how confident you are that
   these regions belong to that network. Say don"t know if you are unsure. Here is the list of regions:
    """

def main() -> None:
    """
    Main function 

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    gemini_key = decouple.config('gemini_key')
    comp_def_location = decouple.config("component_def")
    comp_location = get_locations(os.path.join(comp_def_location, "dice_locations.json"))
 
    
    prompt_str = prompt()
    component_def = {}
    client = genai.Client(api_key=gemini_key)

    print("Quering GEMINI")
    for comp in tqdm(range(len(comp_location.keys())), colour="magenta", unit=" Component"):
        print(f"Comp {comp}")
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt_str + f"{comp_location[str(comp)]}")
        try:
            json_dict = dict(json.loads(response.text.replace("```", "").replace("json", "")))
            json_dict['region'] = comp_location[str(comp)]
            print(f"comp {comp} most likely:")
            print(json_dict['network'])
            print("with ", json_dict['probability'], "probability")
            component_def[f'comp_{comp}'] = json_dict
        except Exception as e:
            print("No response due to ", e)
        print("sleeping for 60 seconds")
        time.sleep(30)
        
    with open(os.path.join(comp_def_location, "gemini_defintion.json"), "w") as f:
        json.dump(component_def, f, indent=4)
    
if __name__ == "__main__":
    main()