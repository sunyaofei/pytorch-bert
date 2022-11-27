

if __name__ == '__main__':
    # https://github.com/tqdm/tqdm
    from tqdm import tqdm
    import time
    for i in tqdm(range(10000)):
        time.sleep(0.01)
        pass