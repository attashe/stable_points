import numpy as np
from utils import Angle


def interpolate_int(start, end, steps):
    return list(np.linspace(start, end, steps+2).astype(int))

def interpolate_float(start, end, steps):
    return list(np.linspace(start, end, steps+2))

def interpolate_str(start, end, steps):
    first = steps // 2
    return [start,] * (first + 1) + [end,] * (steps - first + 1)

def interpolate_vec(start, end, steps):
    raise Exception('Not implemented error')

def interpolate_angle(start, end, steps):
    raise Exception('Not implemented error')


class Interpolator:
    
    def interp(start, end, steps):
        yield start
        
        all_states_dict = {}
        
        for k in start.keys():
            vs = start[k]
            ve = end[k]
            assert type(vs) == type(ve), 'Type of start and end values must be same'
            
            if vs == ve:
                interps = [vs,] * (steps + 2) 
            # TODO: test with different int types
            if type(vs) is int:
                interps = interpolate_int(vs, ve, steps)
            elif type(vs) is float:
                interps = interpolate_float(vs, ve, steps)
            elif type(vs) is str:
                interps = interpolate_str(vs, ve, steps)
            elif type(vs) is Angle:
                interps = interpolate_angle(vs, ve, steps)
            else:
                raise Exception('Not implemented error')

            all_states_dict[k] = interps
        
        print('All different states:')
        print(all_states_dict)
        
        state = start.copy()
        for i in range(steps):
            for k in state.keys():
                state[k] = all_states_dict[k][i+1]
            
            yield state
        
        yield end
