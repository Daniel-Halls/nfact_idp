import numpy as np

def calculate_effect_size(mean, y_sigma):
    return mean/y_sigma

def define_rope_width(y_sigma):
    return  0.1 * y_sigma.mean().item() 

def calculate_rope(y_sigma, posterior):
    rope_width = define_rope_width(y_sigma)
    param = posterior.values.flatten()

    inside_rope = np.mean((param > -rope_width) & (param < rope_width))
    return {
        "inside_rope": inside_rope,
        "outside_rope": 1 - inside_rope
    }
def diagnostics(posterior, posterior_to_check):
    rope = calculate_rope(posterior['value_sigma'], posterior[posterior_to_check])
    print(rope['inside_rope']*100, "Inside ROPE")
    print(rope['outside_rope']*100, "Outside of ROPE")
    print(calculate_effect_size( posterior[posterior_to_check].mean().item(), posterior['value_sigma'].mean().item()), " effect size")