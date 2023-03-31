import threading
import tracemalloc
import time


class Memory(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        tracemalloc.start()

        start_time = time.time()

        while True:
            while time.time() - start_time > 3:
                snapshot = tracemalloc.take_snapshot()

                top_stats = snapshot.statistics('lineno')

                print("[ Top 10 ]")
                for stat in top_stats[:10]:
                    print(stat)

                break


if __name__ == '__main__':
    memory = Memory()
    memory.start()

