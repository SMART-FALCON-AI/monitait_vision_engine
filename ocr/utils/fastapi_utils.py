import subprocess


def is_server_active():
    return {'status': 'server is active'}


def ping(server='example.com', count=1, wait_sec=1):
    """

    :rtype: dict or None
    """
    cmd = "ping -c {} -W {} {}".format(count, wait_sec, server).split(' ')
    try:
        output = subprocess.check_output(cmd).decode().strip()
        lines = output.split("\n")
        total = lines[-2].split(',')[3].split()[1]
        loss = lines[-2].split(',')[2].split()[0]
        timing = lines[-1].split()[3].split('/')
        return {'ping' : {
            'type': 'rtt',
            'min': timing[0],
            'avg': timing[1],
            'max': timing[2],
            'mdev': timing[3],
            'total': total,
            'loss': loss,
        }
        }
    except Exception as e:
        return {'error': str(e)}
