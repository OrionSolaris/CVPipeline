import random
from colorama import Fore
from rich import print


def get_classes(subset, full_list: list):
    if subset:
        subset_string = list(map(full_list.__getitem__, list(map(int, subset))))
        print(
            f"[green]INFO[/green]: {len(subset_string)} classes currently selected for detector: {subset_string}"
        )
    else:
        subset = list(range(len(full_list)))
        print(
            f"[green]INFO[/green]: No subset selected for detector, using all classes..."
        )
    return list(map(int, subset))
