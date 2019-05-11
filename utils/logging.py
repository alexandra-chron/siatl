import math
import sys
import time


def erase_line():
    sys.stdout.write("\033[K")


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return asMinutes(s), asMinutes(rs)


def progress_bar(percentage, bar_len=20):
    filled_len = int(round(bar_len * percentage))
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    return "[{}]".format(bar)


def epoch_progress(epoch, batch, batch_size, dataset_size, start):
    n_batches = math.ceil(float(dataset_size) / batch_size)
    percentage = batch / n_batches

    # stats = 'Epoch:{}, Batch:{}/{} ({0:.2f}%)'.format(epoch, batch, n_batches,
    #                                                   percentage)
    stats = f'Epoch:{epoch}, Batch:{batch}/{n_batches} ' \
            f'({100* percentage:.0f}%)'
    # stats = f'Epoch:{epoch}, Batch:{batch} ({100* percentage:.0f}%)'

    elapsed, eta = timeSince(start, batch / n_batches)
    time_info = 'Time: {} (-{})'.format(elapsed, eta)

    # clean every line and then add the text output
    # log_output = stats + " " + progress_bar + ", " + time_info

    # log_output = " ".join([stats, time_info])
    log_output = " ".join([stats, progress_bar(percentage), time_info])

    sys.stdout.write("\r \r\033[K" + log_output)
    sys.stdout.flush()
    return log_output
