"""
Read DVS128 IBM Gesture dataset
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

_AEDATV3_HEADER_DTYPE = np.dtype(
    [
        ("eventType", np.uint16),
        ("eventSource", np.uint16),
        ("eventSize", np.uint32),
        ("eventTSOffset", np.uint32),
        ("eventTSOverflow", np.uint32),
        ("eventCapacity", np.uint32),
        ("eventNumber", np.uint32),
        ("eventValid", np.uint32),
    ]
)
_AEDATV3_EVENT_DTYPE = np.dtype(
    [("fdata", np.uint32), ("timestamp", np.uint32),]  # In microsecond
)

_LABELS_DTYPE = np.dtype(
        [
            ("event", np.uint8),
            ("start_time", np.uint32),  # In microsecond
            ("end_time", np.uint32),
        ]
)


_dtype = np.dtype([("x", np.uint16), ("y", np.uint16), ("p", np.bool_), ("ts", np.uint64)])


# RGB characteristics
RED = np.array(((255, 0, 0)), dtype=np.uint8)
GREEN = np.array(((0, 255, 0)), dtype=np.uint8)
WHITE = np.array(((255, 255, 255)), dtype=np.uint8)
BLACK = np.array(((0, 0, 0)), dtype=np.uint8)
GREY = np.array(((220, 220, 220)), dtype=np.uint8)

class DVSSpikeTrain(np.recarray):
    """Common type for event based vision datasets"""

    __name__ = "SparseVisionSpikeTrain"

    def __new__(cls, nb_of_spikes, *args, width=-1, height=-1, duration=-1, time_scale=1e-6, **nargs):
        obj = super(DVSSpikeTrain, cls).__new__(cls, nb_of_spikes, dtype=_dtype, *args, **nargs)
        obj.width = width
        obj.height = height
        obj.duration = duration
        obj.time_scale = time_scale  # dt duration in seconds

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.width = getattr(obj, "width", None)
        self.height = getattr(obj, "height", None)
        self.duration = getattr(obj, "duration", None)
        self.time_scale = getattr(obj, "time_scale", None)

def readAEDAT(file: str) -> np.recarray:
    """
    Parsing is made with the AEDAT 3.1 format
    https://inivation.com/support4/software/fileformat/#aedat-31
    with polarity events as packet data types
    Arguments:
        file {str} -- Complete path to file
    Returns:
        {tuple} -- A tuple of spike data and timestamp.
        The spike data is an array of x, y, and polarity values
    """
    assert os.path.exists(file), "File %s doesn't exist." % file
    with open(file, "rb") as f:
        f_bytes = f.read()
    header, _, data = f_bytes.partition(b"\r\n#!END-HEADER\r\n")
    assert data, "Error loading data from file %s" % file
    version, data_format, source, date, *_ = header.split(b"\r\n")
    assert version == b"#!AER-DAT3.1", "Unsupported data format detected"
    offset = 0
    packets = []
    while offset < len(data):
        packet_header = np.frombuffer(
            data, dtype=_AEDATV3_HEADER_DTYPE, count=1, offset=offset
        )
        offset += packet_header.nbytes
        assert (
            packet_header["eventNumber"]
            == packet_header["eventCapacity"]
            == packet_header["eventValid"]
        ), "Something went wrong parsing the event header; your data might be corrupted"
        assert (
            packet_header["eventSize"] == _AEDATV3_EVENT_DTYPE.itemsize
        ), "Packet size doesn't correspond to underlying datatype"
        assert packet_header["eventType"] == 1  # Polarity events
        nb_packets = int(packet_header["eventNumber"])
        packets_data = np.frombuffer(
            data, dtype=_AEDATV3_EVENT_DTYPE, count=nb_packets, offset=offset
        )
        offset += packets_data.nbytes
        packets += packets_data.tolist()

    fdatas, timestamps = np.array(packets).T
    data = DVSSpikeTrain(fdatas.size)
    data.x = np.bitwise_and(np.right_shift(fdatas, 17), 0x7FFF)
    data.y = np.bitwise_and(np.right_shift(fdatas, 2), 0x7FFF)
    data.p = np.bitwise_and(np.right_shift(fdatas, 1), 0x1)
    data.ts = timestamps
    return data


def _read_labels(file: str) -> np.array:
    assert os.path.exists(file), "File %s doesn't exist" % file
    return np.genfromtxt(file, delimiter=",", skip_header=1, dtype=_LABELS_DTYPE)

def aedat2pt(files, save_root, label_map, num_frames=30):
    for ii, filename in enumerate(files):
        labels = _read_labels(filename.replace(".aedat", "_labels.csv"))
        multilabel_spike_train = readAEDAT(filename)

        for jj, (label_id, start_time, end_time) in enumerate(labels):
            label_ = label_map[label_id]
            
            save_dir = os.path.join(save_root, label_)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            event_mask = (multilabel_spike_train.ts >= start_time) & (multilabel_spike_train.ts < end_time)
            ts = multilabel_spike_train.ts[event_mask] - start_time
            x = multilabel_spike_train.x[event_mask]
            y = multilabel_spike_train.y[event_mask]
            p = multilabel_spike_train.p[event_mask]

            coords = np.stack((x, y), axis=-1)
            frames = get_frames(coords=coords, time=ts, polarity=p, num_frames=num_frames)
            frames = torch.from_numpy(frames)
            fname = "/event_{}_{}.pt".format(ii, jj)

            print("Class={}[{}]; Event shape={}; fname={}".format(label_, label_id, list(frames.size()), fname))
            torch.save(frames, save_dir+fname)

def get_frames(
        coords,
        time,
        polarity=None,
        dt=None,
        num_frames=None,
        shape=None,
        flip_up_down=False
    ):
        r"""
        convert the events to rgb frames
        """
        assert time.size > 0, "the length of the time sequence must greater than 0!"
        t_start = time[0]
        t_end = time[-1]
        
        if dt is None:
            dt = int((t_end - t_start) // (num_frames - 1))
        else:
            num_frames = (t_end - t_start) // dt + 1
            
        if shape is None:
            shape = np.max(coords, axis=0)[-1::-1] + 1    # [-1::-1] quickly reverse
            max_val = np.max(coords, axis=0)
        else:
            shape = shape[-1::-1]

        frame_data = np.zeros((num_frames, *shape, 3), dtype=np.uint8)

        if polarity is None:
            colors = GREY
        else:
            colors = np.where(polarity[:, np.newaxis], RED, GREEN)
        
        i = np.minimum((time-t_start) // dt, num_frames - 1)
        x, y = coords.T
        
        frame_data[(i, y, x)] = colors
        if flip_up_down:
            y = shape[0] - y - 1
        return frame_data

if __name__ == '__main__':
    _GESTURE_MAPPING_FILE = "gesture_mapping.csv"
    _TRAIN_TRIALS_FILE = "trials_to_train.txt"
    _TEST_TRIALS_FILE = "trials_to_test.txt"
    root = "/home/jmeng15/data/IBM_Gesture/DVSGesturedataset/DvsGesture/"
    mode = "train"

    # mapping file
    parsed_csv = np.genfromtxt(
            os.path.join(root, _GESTURE_MAPPING_FILE),
            delimiter=",",
            skip_header=1,
            dtype=None,
            encoding="utf-8",
        )
    gestures, indexes = list(zip(*parsed_csv))
    _GESTURE_MAP = dict(zip(indexes, gestures))

    with open(os.path.join(root, _TRAIN_TRIALS_FILE), "r") as f:
        _TRAIN_FILES = map(lambda d: os.path.join(root, d.rstrip()), f.readlines())


    with open(os.path.join(root, _TEST_TRIALS_FILE), "r") as f:
        _TEST_FILES = map(lambda d: os.path.join(root, d.rstrip()), f.readlines())

    _TRAIN_FILES = list(filter(lambda f: os.path.isfile(f), _TRAIN_FILES))
    _TEST_FILES = list(filter(lambda f: os.path.isfile(f), _TEST_FILES))

    save_root = "/home/jmeng15/data/ibm_gesture_pt/"

    if mode == "train":
        FILES = _TRAIN_FILES
    elif mode == "test":
        FILES = _TEST_FILES

    save_root = os.path.join(save_root, mode)
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    
    print(save_root)

    aedat2pt(FILES, save_root=save_root, label_map=_GESTURE_MAP)

    

    # for i, e in enumerate(event):
    #     x, y, time, p, label_id = e
    #     coords = np.stack((x, y), axis=-1)
    #     frames = get_frames(coords=coords, time=time, polarity=p, num_frames=30)
    #     print(label_id)
        
    #     if i == 0:
    #         for t in range(frames.shape[0]):
    #             img = frames[t]
    #             plt.figure(figsize=(10,10))
    #             plt.imshow(img)
    #             plt.savefig(f"./imgs/event{i}{t}.png")
    #             plt.close()