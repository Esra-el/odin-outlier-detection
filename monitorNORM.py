import csv
from datetime import datetime
import psutil
import time

# Library for getting indicators -> https://psutil.readthedocs.io/en/latest/

# ENV vars
CSV_FILENAME = "normal_run.csv"   # raccolta SENZA injection, usata dalla Prova 3
SAMPLING_INTERVAL = 1  # secondi tra una lettura e l'altra
DURATION_SEC = 14100    # durata della raccolta in secondi (3600 = 1 ora); None = senza limite
                       # questo file serve solo per la raccolta normale: NON avviate l'injector


def current_milli_time():
    """
    This function returns the current time in milliseconds
    """
    return round(time.time() * 1000)


def read_data():
    """
    This function reads data from the system
    :return: a dictionary
    """
    data_dict = {"time": current_milli_time(),
                 "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    cpu_probe(data_dict)
    vm_probe(data_dict)
    io_probe(data_dict)
    return data_dict


def cpu_probe(data_dict):
    """
    This function reads CPU data from the system and uses it to update a dict
    """
    # Percentuale di CPU usata dall'ultima chiamata (cioe' nell'ultimo secondo)
    data_dict["cpu_percent"] = psutil.cpu_percent()

    # Tempi cumulativi della CPU (codice originale del prof)
    cpu_t = psutil.cpu_times()
    data_dict["idle_time"] = cpu_t.idle
    data_dict["usr_time"] = cpu_t.user
    # 'interrupt' esiste solo su Windows: su Mac/Linux scrive 0 invece di andare in errore
    data_dict["interrupt_time"] = getattr(cpu_t, "interrupt", 0)


def vm_probe(data_dict):
    """
    This function reads VM data from the system and uses it to update a dict
    """
    vm_data = psutil.virtual_memory()
    data_dict["mem_total"] = vm_data.total
    data_dict["mem_available"] = vm_data.available
    data_dict["mem_percent"] = vm_data.percent


# Ultime letture dei contatori di disco e rete: servono per calcolare quanto sono
# aumentati nell'ultimo secondo (i contatori di psutil crescono sempre dall'accensione)
last_io = {"disk": None, "net": None, "time": None}


def io_probe(data_dict):
    """
    This function reads disk and network data and stores how much was read/written
    and received/sent per second since the previous reading
    """
    disk = psutil.disk_io_counters()
    net = psutil.net_io_counters()
    now = time.time()
    if last_io["time"] is None:
        # prima lettura: non c'e' ancora un valore precedente
        elapsed = None
    else:
        elapsed = now - last_io["time"]
    if elapsed:
        data_dict["disk_read_mb_s"] = round((disk.read_bytes - last_io["disk"].read_bytes) / 1e6 / elapsed, 3)
        data_dict["disk_write_mb_s"] = round((disk.write_bytes - last_io["disk"].write_bytes) / 1e6 / elapsed, 3)
        data_dict["net_recv_kb_s"] = round((net.bytes_recv - last_io["net"].bytes_recv) / 1e3 / elapsed, 3)
        data_dict["net_sent_kb_s"] = round((net.bytes_sent - last_io["net"].bytes_sent) / 1e3 / elapsed, 3)
    else:
        data_dict["disk_read_mb_s"] = 0.0
        data_dict["disk_write_mb_s"] = 0.0
        data_dict["net_recv_kb_s"] = 0.0
        data_dict["net_sent_kb_s"] = 0.0
    last_io["disk"], last_io["net"], last_io["time"] = disk, net, now


def write_dict_to_csv(filename, dict_item, is_first_time):
    """
    This function writes a dictionary as a row of a CSV file
    """
    mode = 'w' if is_first_time else 'a'
    with open(filename, mode, newline="") as f:
        w = csv.DictWriter(f, dict_item.keys())
        if is_first_time:
            w.writeheader()
        w.writerow(dict_item)


if __name__ == "__main__":
    """
    Main of the monitor
    """
    # Prima chiamata "a vuoto": cpu_percent() la prima volta restituisce sempre 0.0,
    # e i contatori di disco e rete hanno bisogno di una lettura di partenza
    psutil.cpu_percent()
    io_probe({})
    time.sleep(SAMPLING_INTERVAL)

    is_first_time = True
    start = time.time()
    try:
        # continua finche' non e' passato DURATION_SEC (o per sempre, se DURATION_SEC = None)
        while DURATION_SEC is None or time.time() - start < DURATION_SEC:
            monit_data = read_data()
            print(monit_data)
            write_dict_to_csv(CSV_FILENAME, monit_data, is_first_time)
            is_first_time = False
            time.sleep(SAMPLING_INTERVAL)
        print("\nRaccolta finita dopo", DURATION_SEC, "secondi. Dati salvati in", CSV_FILENAME)
    except KeyboardInterrupt:
        print("\nMonitor fermato. Dati salvati in", CSV_FILENAME)