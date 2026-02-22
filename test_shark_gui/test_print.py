import os
import time


if __name__ == '__main__':

    for i in range(8):
        print(f'test text: {i}')
        time.sleep(1)
        os.system('cls')
        # print('\033c', end='')
        time.sleep(1)
