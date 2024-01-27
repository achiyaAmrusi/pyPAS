# this needs to be a coincidence filter module where the filter needs to be manageable object
# this object suppose to get the detectors and the time stamps and with that give the coincidence cases
# later we can add another restriction on the filter
from .pas_cdb_filter import cdb_pairs_filter_from_dataframe, cdb_pairs_filter_from_files
