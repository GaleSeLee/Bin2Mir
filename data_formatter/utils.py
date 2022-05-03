from collections import defaultdict


class DefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def statistic_dict(labels: list, infos: list) -> dict:
    cnt_dict = {label: 0 for label in set(labels)}
    info_dict = {label: [] for label in set(labels)}
    for label, info in zip(labels, infos):
        cnt_dict[label] += 1
        info_dict[label].append(info)
    return {
        'cnt_dict': cnt_dict,
        'info_dict': info_dict
    }

