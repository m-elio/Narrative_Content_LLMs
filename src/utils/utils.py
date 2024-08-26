import os
import inspect

import src.input_format.format as format_module

def print_on_main(to_print):
    if os.environ.get('RANK', '0') == '0' and os.environ.get('LOCAL_RANK', '0') == '0':
        print(to_print)

available_formats = {obj.format_name: obj
                     for _, obj in inspect.getmembers(format_module, inspect.isclass)
                     if issubclass(obj, format_module.Format) and not obj == format_module.Format}

def print_setup_decorator(string_init):

    def decorator(function):

        def wrapper(*args, **kwargs):
            print("*" * 80)
            print(f"{string_init}")
            print("-" * 80)
            print()

            output = function(*args, **kwargs)

            print("Step completed!")
            print()
            print("*" * 80)
            print()

            return output

        return wrapper

    return decorator
