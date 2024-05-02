import os
import multiprocessing
import subprocess


def process_products(pid):
    subprocess.run(['./run_web_agent_site_env.sh', f"{pid}"])


def main():
    num_pids = 1
    processes = []
    for i in range(1, num_pids + 1):
        process = multiprocessing.Process(target=process_products, args=([i]))
        processes.append(process)
        process.start()
    
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()