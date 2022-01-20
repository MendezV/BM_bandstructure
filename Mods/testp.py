import multiprocessing as mp
import numpy as np
import time
import concurrent.futures
def my_function2( args):
    (i, param1, param2, param3)=args
    result = param1 ** 2 * param2 + param3
    time.sleep(2)
    print("ds")
    return (i, result)

def my_function(i, param1, param2, param3):
    result = param1 ** 2 * param2 + param3
    time.sleep(2)
    print("ds")
    return (i, result)


def get_result(result):
    global results
    results.append(result)


if __name__ == '__main__':
    print("ds")
    params = np.random.random((10, 3)) * 100.0
    results = []
    # ts = time.time()
    # for i in range(0, params.shape[0]):
    #     get_result(my_function(i, params[i, 0], params[i, 1], params[i, 2]))
    # print('Time in serial:', time.time() - ts)
    # print(results)

    results = []
    ts = time.time()
    pool = mp.Pool(mp.cpu_count())
    for i in range(0, params.shape[0]):
        pool.apply_async(my_function, args=(i, params[i, 0], params[i, 1], params[i, 2]), callback=get_result)
    pool.close()
    pool.join()
    print('Time in parallel:', time.time() - ts)
    print(results)
    
    MAX_WORKERS=10
    ts = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
                executor.submit(my_function2, args=(i, params[i, 0], params[i, 1], params[i, 2])): i for i in range(0, params.shape[0])
            }

        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()  # read the future object for result
    print('Time in parallel 2:', time.time() - ts)
    print(results)
                