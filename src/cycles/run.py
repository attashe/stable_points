"""
Пакет для генерации промежуточных кадров между двумя положениями камер
Содержит инструменты для автоматизации основного приложения и систему команд для
записи скриптов для зацикливания алгоритма обработки промежуточных кадров.

Файл utility.py содержит вспомогательные функции для интерполяции движения камеры и других
характеристик.
"""
from action import Action, ActionBuilder
from scenario import Scenario
from scenario_runner import ScenarioRunner


def func_1(state: dict):
    x = state['x']
    state['y'] = x * state['a']
    return state

def func_2(state: dict):
    state['result'] = state['x'] + state['y']
    return state

def main():
    action1 = Action(func_1)
    action2 = Action(func_2)

    init_params = {
        'a': 5,
        'b': 10,
    }

    end_params = {
        'a': 10,
        'b': 1,
    }

    init_state = {
        'x': 0,
        'y': 0,
        'result': 0
    }

    steps = 3

    scenario = Scenario([action1, action2])

    scenario_runner = ScenarioRunner(scenario)

    result_state = scenario_runner.run(init_params, end_params, init_state, steps)
    
    print(result_state)

if __name__ == "__main__":
    main()
