import multiprocessing
import time


def test0(event):
    event.clear()
    print(f"{multiprocessing.current_process().name}_start========")
    event.set()
    time.sleep(5)
    event.clear()
    print(f"{multiprocessing.current_process().name}_end========")
    event.set()


def test1(event):
    print(f"{multiprocessing.current_process().name}_start========")
    event.wait()
    print(f"{multiprocessing.current_process().name}_end========")


if __name__ == '__main__':
    event = multiprocessing.Event()
    task0 = multiprocessing.Process(target=test0, args=(event,), name="Pro_test0")
    task1 = multiprocessing.Process(target=test1, args=(event,), name="Pro_test1")

    task0.start()
    task0.join()

    task1.start()
    task1.join()


"""
运行结果：
Pro_test0_start========
Pro_test0_end========
Pro_test0_end========

Process finished with exit code 0
"""
