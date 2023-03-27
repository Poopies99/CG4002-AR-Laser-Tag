import threading
import time

class Test(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        i = 0

        start = time.time()

        while True:
            i += 1

            if i == 1000000:
                break

        end = time.time()

        print(end - start)

if __name__ == '__main__':
    test = Test()

    test.start()