import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table


class Timer:
    def __init__(self):

        self.read_frames = {"name": "Read Frame", "values": np.array([])}
        self.send_recieve = {"name": "Send Recieve", "values": np.array([])}
        self.show_frames = {"name": "Show Frame", "values": np.array([])}

        self.total_serialise = {
            "name": "Total Serialisation + Send",
            "values": np.array([]),
        }
        self.serialise = {
            "name": "Serialisation + Send (Per Component)",
            "values": np.array([]),
        }

        self.det_preproc = {"name": "Detector Preprocessing", "values": np.array([])}
        self.det_infer = {"name": "Detector Inference", "values": np.array([])}
        self.det_postproc = {"name": "Detector Postprocessing", "values": np.array([])}

    def print_avgs(self):
        table = Table(title="Average Time for Each Section")
        table.add_column("Section", style="magenta")
        table.add_column("Time Taken", justify="right", style="blue")
        for attribute, value in self.__dict__.items():
            table.add_row(
                value.get("name"), f"{round(np.average(value.get('values')[1:]), 1)}ms"
            )
        console = Console()
        console.print(table)
