import os
import time

def disk_io_benchmark(file_size_mb=1024):
    filename = 'testfile.tmp'
    data = os.urandom(1024 * 1024)  # 1MB block
    start = time.time()
    
    with open(filename, 'wb') as f:
        for _ in range(file_size_mb):
            f.write(data)

    write_time = time.time() - start
    print(f'Wrote {file_size_mb}MB in {write_time:.2f} seconds ({file_size_mb/write_time:.2f} MB/s)')

    start = time.time()
    with open(filename, 'rb') as f:
        while f.read(1024 * 1024):
            pass

    read_time = time.time() - start
    print(f'Read {file_size_mb}MB in {read_time:.2f} seconds ({file_size_mb/read_time:.2f} MB/s)')

    os.remove(filename)

if __name__ == '__main__':
    disk_io_benchmark()
