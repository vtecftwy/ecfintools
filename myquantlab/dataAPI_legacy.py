"""
TODO: add docstring for script file
"""

import collections
import configparser
import http.client
import json
import logging
import os
import random
import re
import socket
import ssl
import time
from abc import ABC, abstractmethod
from datetime import datetime
from datetime import time as tm
from datetime import timedelta, tzinfo
from functools import partial
from pathlib import Path
from urllib import parse as urlparse
from typing import Optional

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
import requests
from dateutil import tz
from IPython.display import display
from pandas.tseries.holiday import (AbstractHolidayCalendar, EasterMonday,
                                    GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay,
                                    USPresidentsDay, USThanksgivingDay,
                                    nearest_workday, next_monday, next_workday)
from pandas.tseries.offsets import (BusinessDay, BusinessHour,
                                    CustomBusinessDay, CustomBusinessHour,
                                    DateOffset, Day, Easter, Hour, Week)

from myquantlab import PACKAGE_ROOT

""" Data source info for API not implemented:"""
""" 
'xm-com': {'name': 'xm-com',
           'directory': 'xm-com-mt4',
           'format': 'mt4'},
'metatrader': {'name': 'metatrader',
               'directory': 'metatrader-mt4',
               'format': 'mt4'},
'mixed-mt4': {'name': 'mixed-mt4',
              'directory': 'mixed-mt4',
              'format': 'mt4'},
'wsj': {'name': 'wsj',
        'directory': 'wsj',
        'format': 'wsj'}
"""


def _running_in_ipython():
    try:
        __IPYTHON__ # type: ignore
        code_running_in_ipython = True
    except NameError:
        code_running_in_ipython = False
    return code_running_in_ipython


RUNNING_IN_IPYTHON = _running_in_ipython()

if RUNNING_IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class MT4ServerTime(tzinfo):
    """Class for MT4 Server Time timezone.

    By default, MT4 Server Time is set as UTC+2 when NY is in standard time and UTC+3 when NY is in day saving time.
    The actual offset can be changed.

    Attributes:
        hours (int):    Number of hours offset from UTC for MT4 when not in DST. Default is 2
        reprname (str): Timezone name when __repr__ is called
        stdname (str):  Timezone name returned by .tzname when in standard time
        dstname (str):  Timezone name returned by .tzname when in day saving time

    Based on USTimeZone as in https://docs.python.org/3/library/datetime.html#tzinfo-objects
    Detailed documentation: # https://github.com/pandas-dev/pandas/blob/master/pandas/_libs/tslibs/timezones.pyx
    """

    ZERO = timedelta(0)
    HOUR = timedelta(hours=1)
    SECOND = timedelta(seconds=1)

    def __init__(self, hours=2, reprname="MT4 Server Time", stdname="MT4 Std", dstname="MT4 Dst", verbose=False):
        """Initialize class MT4ServerTime

        Attributes:
            hours (int):    Offset to UTC in hours. Default = 2
            reprname (str): Name of the timezone for printing. Default: 'MT4 Server Time"
            stdname (str):  Name of the timezone when in standard time. Default: 'MT4 Std'
            dstname (str):  Name of the timezone when in standard time. Default: 'MT4 Dst'
        """
        self.stdoffset = timedelta(hours=hours)
        self.reprname = reprname
        self.stdname = stdname
        self.dstname = dstname
        self.verbose = verbose
        if self.verbose: print('>>> MT4 Server Class Initialized in Verbose Mode.')

    def __repr__(self):
        return self.reprname

    def tzname(self, dt):
        if self.dst(dt):
            # self.printlog(f"tzname(): dst branch")
            return self.dstname
        else:
            # self.printlog(f"tzname(): std branch")
            return self.stdname

    def utcoffset(self, dt):
        """Return offset of local time from UTC, as a timedelta object that is positive east of UTC."""
        return self.stdoffset + self.dst(dt)

    def dst(self, dt):
        """Return the daylight saving time (DST) adjustment, as a timedelta object or None if DST info isn’t known.
        
        Defines STD and DST periods by `start` and `end` or DST period (each are naive `datetime` objects):
            `start`, `end` and `dt` lead to four possibilities: Standard time, Day saving time, Gap or Ambiguous:

                              start ->|<-  1Hr  ->|                      end-1 hour ->|<-  1Hr  ->|
            STD <=====================|           |                                   |============================>
            DST                       |           |===============================================|
                                      |           |<- start + 1 hour                  |           |<- end
                Std Time in effect    |    Gap    |     Day Saving Time in Effect     | Ambiguous | Standard Time in effect

            fold allows to decide for the Ambiguous case
        """
        if dt is None:
            # An exception may be sensible here, in one or both cases. It depends on how you want to treat them.  
            # The default `.fromutc()` implementation (called by the default `.astimezone()` implementation) 
            # passes a `datetime` with `dt.tzinfo` as `self`.
            return self.ZERO

        if dt.tzinfo is None:
            # In case `dt` is naive (no tzinfo), set it as a the current time zone
            dt = dt.replace(tzinfo=self)
        assert dt.tzinfo is self, f"dt.tzinfo is not the same as the class: {dt.tzinfo} instead of {self.reprname}"
        start, end = self.us_dst_range(dt.year)

        # Can't compare naive (start and end) to aware objects (dt), so strip the timezone from dt first.
        dt = dt.replace(tzinfo=None)
        if start + self.HOUR <= dt < end - self.HOUR:
            # DST is in effect.
            return self.HOUR
        elif start <= dt < start + self.HOUR:
            # Gap (a non-existent hour): reverse the fold rule.
            return self.HOUR if dt.fold else self.ZERO
        elif end - self.HOUR <= dt < end:
            # Fold (an ambiguous hour): use dt.fold to disambiguate.
            return self.ZERO if dt.fold else self.HOUR
        else:
            # DST is off.
            return self.ZERO

    # def fromutc(self, dt):
    #     """This is called from the default datetime.astimezone() implementation.

    #     When called from that, dt.tzinfo is self, and dt’s date and time data are to be viewed as expressing a UTC time
    #     The purpose of fromutc() is to adjust the date and time data, returning an equivalent datetime in self’s local
    #     time.

    #     Technical Notes:
    #     astimezone()
    #     https://docs.python.org/3/library/datetime.html#datetime.datetime.astimezone
    #     https://github.com/python/cpython/blob/3.10/Lib/datetime.py#L1860

    #     fromutx()
    #     https://docs.python.org/3/library/datetime.html#datetime.tzinfo.fromutc
    #     https://github.com/python/cpython/blob/3.10/Lib/datetime.py#L1137

    #     dateutil.tz
    #     https://github.com/dateutil/dateutil/blob/master/src/dateutil/tz/tz.py#L386
    #     https://github.com/dateutil/dateutil/blob/master/src/dateutil/tz/tz.py#L743
    #     https://github.com/dateutil/dateutil/blob/master/src/dateutil/tz/tz.py#L830

    #     """
    #     assert dt.tzinfo is self, f"dt.tzinfo is not the same as the the class: {dt.tzinfo} instead of {self.reprname}"
    #     start, end = self.us_dst_range(dt.year)
    #     start = start.replace(tzinfo=self)
    #     end = end.replace(tzinfo=self)
    #     std_time = dt + self.stdoffset
    #     dst_time = std_time + self.HOUR

    #     if end <= dst_time < end + self.HOUR:
    #         # Repeated hour
    #         return std_time.replace(fold=1)
    #     if std_time < start or dst_time >= end:
    #         # Standard time
    #         return std_time
    #     if start <= std_time < end - self.HOUR:
    #         # Daylight saving time
    #         return dst_time

    @staticmethod
    def us_dst_range(year):
        """Finds start and end times for US DST. For years before 1967, return start = end for no DST.

        US DST Rules
        Simplified (i.e., wrong for a few cases) set of rules for US DST start and end times.
        For a complete and up-to-date set of DST rules and timezone definitions, visit the Olson Database (or try pytz):
        http://www.twinsun.com/tz/tz-link.htm, http://sourceforge.net/projects/pytz/ (might not be up-to-date)
        Since 2007, DST:
         - starts at 2am (standard time) on the second  Sunday in March, which is the first Sunday on or after Mar 8 and
         - ends at 2am (DST time) on the first Sunday of Nov.
        From 1987 to 2006, DST used:
         - to start at 2am (standard time) on the first Sunday in April and
         - to end at 2am (DST time) on the last Sunday of October, which is the first Sunday on or after Oct 25.
        From 1967 to 1986, DST used:
         - to start at 2am (standard time) on the last Sunday in April (the one on or after April 24) and
         - to end at 2am (DST time) on the last Sunday of October, which is the first Sunday on or after Oct 25.
         """
        DSTSTART_2007 = datetime(1, 3, 8, 2)
        DSTEND_2007 = datetime(1, 11, 1, 2)
        DSTSTART_1987_2006 = datetime(1, 4, 1, 2)
        DSTEND_1987_2006 = datetime(1, 10, 25, 2)
        DSTSTART_1967_1986 = datetime(1, 4, 24, 2)
        DSTEND_1967_1986 = DSTEND_1987_2006

        def first_sunday_on_or_after(dt):
            days_to_go = 6 - dt.weekday()
            if days_to_go:
                dt += timedelta(days_to_go)
            return dt

        if 2006 < year:
            dststart, dstend = DSTSTART_2007, DSTEND_2007
        elif 1986 < year < 2007:
            dststart, dstend = DSTSTART_1987_2006, DSTEND_1987_2006
        elif 1966 < year < 1987:
            dststart, dstend = DSTSTART_1967_1986, DSTEND_1967_1986
        else:
            return (datetime(year, 1, 1), ) * 2

        start = first_sunday_on_or_after(dststart.replace(year=year))
        end = first_sunday_on_or_after(dstend.replace(year=year))
        return start, end


class DETradingHolidays(AbstractHolidayCalendar):
    """Custom holiday calendar for Frankfurt financial markets
    
    Note: 
        Use `DETradingHolidays.rules` to see the holiday rules
        Use `DETradingHolidays.holidays(start_date, end_date, return_names=True)` to see dates
    """
    calname = 'Frankfurt Financial Markets'
    rules = [
        Holiday('New Year Day', month=1, day=1, observance=nearest_workday),
        GoodFriday,
        EasterMonday,
        Holiday('Labor Day', month=5, day=1, days_of_week=(0, 1, 2, 3, 4,)),
        Holiday('Whith Monday', month=1, day=1, offset=[Easter(), Day(50)]),
        Holiday("Christmas Eve", month=12, day=24, days_of_week=(0, 1, 2, 3, 4,)),
        Holiday("Christmas Day", month=12, day=25, days_of_week=(0, 1, 2, 3, 4,)),
        Holiday("Boxing Day", month=12, day=26, days_of_week=(0, 1, 2, 3, 4,)),
    ]   


class FRTradingHolidays(AbstractHolidayCalendar):
    """Custom holiday calendar for Paris financial markets
    
    Note: 
        Use `FRTradingHolidays.rules` to see the holiday rules
        Use `FRTradingHolidays.holidays(start_date, end_date, return_names=True)` to see dates
    """
    calname = 'Paris Financial Markets'
    rules = [
        Holiday('New Year Day', month=1, day=1, days_of_week=(0, 1, 2, 3, 4,)),
        GoodFriday,
        EasterMonday,
        Holiday('Labor Day', month=5, day=1, days_of_week=(0, 1, 2, 3, 4,)),
        Holiday("Christmas Day", month=12, day=25, days_of_week=(0, 1, 2, 3, 4,)),
        Holiday("Boxing Day", month=12, day=26, days_of_week=(0, 1, 2, 3, 4,)),
    ]   


class UKTradingHolidays(AbstractHolidayCalendar):
    """Custom holiday calendar for London financial markets
    
    Note: 
        Use `UKTradingHolidays.rules` to see the holiday rules
        Use `UKTradingHolidays.holidays(start_date, end_date, return_names=True)` to see dates
    """
    calname = 'London  Financial Markets'
    rules=[
        Holiday('New Year Day', month=1, day=1, observance=next_monday),
        GoodFriday,
        EasterMonday,
        Holiday(
            'Early May Bank Holiday', month=4, day=30, 
            # First Monday of May (https://www.timeanddate.com/holidays/uk/early-may-bank-holiday)
            offset=Week(weekday=0)
        ),
        Holiday(
            'Spring Bank Holiday', month=6, day=1, 
            # Last Monday of May (https://www.timeanddate.com/holidays/uk/spring-bank-holiday)
            offset= -1 * Week(weekday=0)
        ),
        Holiday(
            'Summer Bank Holiday', month=9, day=1, 
            # Last Monday of Aug (https://www.timeanddate.com/holidays/uk/summer-bank-holiday)
            offset= -1 * Week(weekday=0)
        ),
        Holiday("Christmas Day", month=12, day=25, observance=next_workday),
        Holiday("Boxing Day", month=12, day=26, observance=next_workday),
    ]


class USTradingHolidays(AbstractHolidayCalendar):
    """Custom holiday calendar for US financial markets
    
    Note: 
        Use `USTradingHolidays.rules` to see the holiday rules
        Use `USTradingHolidays.holidays(start_date, end_date, return_names=True)` to see dates
    """
    calname = 'US Financial Markets'
    rules = [
        Holiday('New Year Day', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('Juneteenth Day', month=6, day=19, observance=nearest_workday),
        Holiday('Independance Day', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday)
    ]   


class InfoBox:
    """Class to make printing into a jupyter notebook easier, by using information box widgets

    Represents an output widget from ipywidgets,. Allows to print text in a specific information box, and clear
    it.

    Creation: `box = InfoBox(box_header='Text to add as first line', show_box=True)`


    Attributes:
        `self.box` (widgets.Output):  output widget created for this information box
    """

    def __init__(self, box_header=None, show_box=True):
        """
        Create a new information box

        Class checks that it is running in IPython, or disables displaying the box

        Args:
            box_header (str):   Text to add as first line of the information box
                                Default is None
            show_box (bool):    If True, display the info box directly, otherwise, wait for explicit .show_box() call
        """
        self._in_ipython = self._running_in_ipython()
        self.box = widgets.Output(layout={'border': '1px solid lightgrey'})

        if box_header is not None:
            with self.box:
                print(box_header)

        if show_box and self._running_in_ipython():
            display(self.box)

    def print(self, text=None):
        """
        prints the passed string as next line of the given information box

        Args:
            text (str): string to print
        """
        if self._in_ipython:
            with self.box:
                print(text)

    def clear(self, wait=True):
        """
        clears the text box, immediately or upon next print call, depending of the value od wait

        Args:
            wait (bool):    If true, displays the info box immediately. Otherwise, wait till next print operation

        Returns:

        """
        if self._in_ipython:
            self.box.clear_output(wait=wait)

    def show_box(self):
        """Displays the information box"""
        if RUNNING_IN_IPYTHON:
            display(self.box)

    @staticmethod
    def _running_in_ipython():
        try:
            __IPYTHON__ # type: ignore
            code_running_in_ipython = True
        except NameError:
            code_running_in_ipython = False
        return code_running_in_ipython


class ApiKeyRing(collections.deque):
    """
    Implements a ring serving API keys successively in an infinite loop.

    ApiKeyRing instances are deque objects, initialized with API keys from a *.cfg file, with which the following
    methods can be used:
        ApiKeyRing.get_new_key()
            first "rotates" the ring to expose a new API key, then reads the new 'HEAD' key
        ApiKeyRing.read_key()
            just returns the HEAD key.

    HEAD is the last left element of the underlying deque
    TAIL is the last right element of the underlying deque
    Rotating the ring means popping the left key and then appending it to the right

    ApiKeyRing extends collection.deque. It inherits methods such as append, pop, popleft, ...
    """

    def __init__(self, config:dict, source:str='alphavantage'):
        """
        Initiate ApiKeyRing for the source, using the config dictionary. Default is alphavantage.

        Args:
            config (dict):  dict with key-value pairs for all info in *.cfg files
            source (str):   name of the historical price source
        """
        super(ApiKeyRing, self).__init__()
        key_search_pattern = f"{source}_"
        keys = [key for key in config.keys() if key_search_pattern in key]
        random.shuffle(keys)
        for key in keys:
            self.append(config[key])
        logging.info(f"{len(self)} API keys retrieved for {source}")

    def get_new_key(self):
        """Changes current HEAD key to TAIL and returns the new HEAD"""
        self.rotate(-1)
        return self.read_key()

    def read_key(self):
        """Returns current 'HEAD' key without rotating the ring"""
        return self[0]


class API(ABC):
    """
    Base class from which all financial data source APIs classes are subclassed.

    Represent the respective data source API and gives access to these API

    Attributes:
        source (str):
        folder (str):
        data_format (str):
        drive (str):
        datasets (Path):
        processed (Path):
        raw_data (Path):
        timeframes (str):
        ticker_lists (dict):
        ticker_dictionary (dict):
        config (dict):
        downloads_to_retry_dict (dict):
        failed_downloads_dict (dict):
        instructions_list (list):
        ds_info_dictionary (dict(tuple(date str))):
        api_key_ring:

    Methods:
        info
        info_methods
        switch_to_NAS
        switch_to_HDD
        date_latest_file_download
        get_dataset

    TODO: add all compulsory instance attributes for all classes that must be defined in each API subclass
    """

    # Define a list of standard timeframes for all API. Translated into source specific code in each API
    timeframes = ['M', 'W', 'D', 'H1', 'M30', 'M15', 'M5', 'M1']
    source = ''
    folder = ''
    data_format = None

    def __init__(self):
        self.drive = 'project'
        data_dir:Path = (PACKAGE_ROOT / '../data').resolve()
        self.datasets:Path = data_dir / 'datasets' / self.folder
        self.processed:Path = data_dir / 'processed-data' / self.folder
        self.raw_data:Path = data_dir / 'raw-data' / self.folder

        self.config = self.load_config(Path().cwd())    # Retrieve the config file

        self._create_ticker_dictionary()
        self._create_api_info_dict()
        self.api2source_timeframe_dict = {}
        self._create_api2source_timeframe_dict()
        self._create_source2ticker_name()
        self._define_ticker_lists()
        self._define_ds_info_dictionary()

        self.api_key_ring = None
        self.current_ticker = None
        self.current_tf = None
        self.adjusted = True
        self.methods = []
        self.downloads_to_retry_dict = {}
        self.failed_downloads_dict = {}
        self.instructions_list = []

    def info(self):
        """Prints key info about the current instance of data source object's attributes"""
        # print(f"0        1         2         3         4         5         6         7         8         9         ")
        # print(f"123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789")
        print(f"{40 * '='} Key Instance Attributes {40 * '='}")
        print(f"source: .......... {self.source}")
        print(f"drive: ........... {self.drive}")
        print(f"raw_data ......... {self.raw_data.absolute()}")
        print(f"datasets Folder .. {self.datasets.absolute()}")
        print(f"data_format ...... {self.data_format}")
        print(f"timeframes ....... {self.timeframes}")
        print(f"{40 * '='} Other Information {40 * '='}")
        if self.api_key_ring is not None:
            print(f"Number API Keys .. {len(self.api_key_ring)}")
        else:
            print(f"No API key for {self.source}")
        print(f"Ticker lists and respective number of tickers:")
        for k, l in self.ticker_lists.items():
            print(f"  - ticker_list['{k}'] has {len(l)} tickers")
        print(f"Date of last file downloaded: {self.date_latest_file_download():%Y-%m-%d}")

    def info_methods(self, show_docstrings=True):
        """Prints the list of available methods with their respective docstrings

        Parameters
        ----------
        show_docstrings : bool, optional
            If True, shows the docstrings with the list of methods, by default True
        """
        # print(f"0        1         2         3         4         5         6         7         8         9         ")
        # print(f"123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789")
        print(f"{30 * '='} {'Available methods with docstrings':^38s} {30 * '='}")
        for i, fct in enumerate(self.methods, 1):
            print(f"{i: 2d}. {fct.__name__}():")
            if show_docstrings:
                print(f"    {fct.__doc__}")
                print(f"{90 * '-'}")

    def switch_to_NAS(self):
        """Set datasource object to use NAS drive"""
        self.drive = 'NAS'
        self.datasets = Path('R:\\Financial Data\\Historical Prices Datasets') / self.folder
        self.raw_data = Path('R:\\Financial Data\\Historical Prices Raw Data') / self.folder
        logging.info('Switched drive to NAS folder')

    def switch_to_HDD(self):
        """Set datasource object to use HDD a.k.a. project drive"""
        self.drive = 'project'
        self.datasets = (PACKAGE_ROOT / '../data/datasets' / self.folder).resolve()
        self.raw_data = (PACKAGE_ROOT / '../data/raw-data' / self.folder).resolve()
        logging.info('Switched drive to HDD project folder')

    def load_config(self, cwdir:Path)->dict:
        """Load the config file including API keys and other configuration parameters
        FIXME: Modify to get config file on local computer (see metagentools), or PACKAGE_ROOT
        """

        p2cfg = (PACKAGE_ROOT / '../config-api-keys.cfg').resolve()
        configuration = configparser.ConfigParser()
        configuration.read(p2cfg)
        config_dict = {}
        for section in configuration:
            for key in configuration[section]:
                config_dict[key] = configuration[section][key]
        return config_dict

    def _create_api_info_dict(self):
        """Creates API information dictionary including API data such as last call to API, ...

        Load API info from json file and creates the attribute api_info_dict including info for current source

        Default JSON format (may be different for each source based on the API):
        "api_source_name": {
                        "api_quota": int,                       quota in number of calls
                        "api_quota_period": int,                period in hours over which quota is computed
                        "api_last_call_datetime: datetime,      date time in timezone of API
                        "api_current_period_total_calls" int,   qty (count of API made during the current period
                        "api_max_frequency": int,               qty
                        "api_max_frequency_time: int;           seconds
                        }
        """
        # TODO: move the name of the json file to a CONSTANT or in a user setting
        p2json_file = PACKAGE_ROOT / 'assets/json/API_info.json'
        if p2json_file.is_file():
            with open(p2json_file, 'r') as fp:
                json_file = json.load(fp)
            self.api_info_dict = json_file.get('default')
        else:
            raise ValueError(f"No json file with name <{p2json_file.name}> at {p2json_file.absolute()}")

    def _create_source2ticker_name(self):
        """
        Create translation dictionary between data source ticker names and ticker_dictionary names for the source API

        Uses the values in self.ticker_dictionary
        """
        self.source2ticker_name = dict([(subdict['tickers'][self.source], symbol)
                                        for symbol, subdict in self.ticker_dictionary.items()
                                        if self.source in subdict['tickers'].keys()])

    def _create_ticker_dictionary(self):
        """
        Creates the instance attribute ticker_dictionary

        self.ticker_dictionary is created by loading json file
        """
        p2json_file = PACKAGE_ROOT / 'assets/json/historical_prices_tickers.json'
        if p2json_file.is_file():
            with open(p2json_file, 'r') as f:
                self.ticker_dictionary = json.load(f)
        else:
            msg = f"No json file with name <{p2json_file.name}> at {p2json_file.absolute()}"
            raise ValueError(msg)

    def _define_ds_info_dictionary(self):
        """
        Sets instance attribute self.ds_info_dictionary

        Loads dataset info dictionary from dataset_info.json file
        Checks that ticker info in dataset_info and in historical_prices_ticker json files are consistent
        Technical note: use pkg_resources to point to package directory
        """

        p2json_file = PACKAGE_ROOT / 'assets/json/datasets_info.json'
        if p2json_file.is_file():
            with open(p2json_file, 'r') as f:
                self.ds_info_dictionary = json.load(f)
        else:
            msg = f"No json file with name <{p2json_file.name}> at {p2json_file.absolute()}"
            raise ValueError(msg)

        # test `ds_info_dictionary` consistency with `ticker_dictionary`
        tickers_keys = set(self.ticker_dictionary.keys())
        ds_info_dict_keys = set(self.ds_info_dictionary.keys())

        in_ticker_not_in_ds = tickers_keys.difference(ds_info_dict_keys)
        in_ds_info_not_in_ticker = ds_info_dict_keys.difference(tickers_keys)
        if in_ticker_not_in_ds != set() or in_ds_info_not_in_ticker != set():
            print('historical_prices_tickers.json and datasets_info.json are not consistent')
            print('Tickers in historical_prices_tickers.json and not in datasets_info.json:')
            print(in_ticker_not_in_ds)
            for t in in_ticker_not_in_ds:
                print(f"{t} {'-'*80}")
                display(self.ticker_dictionary[t])
            print('Tickers in datasets_info.json and not in historical_prices_tickers.json:')
            print(in_ds_info_not_in_ticker)
            for t in in_ds_info_not_in_ticker:
                print(f"{t} {'-'*80}")
                display(self.ds_info_dictionary[t])
            raise RuntimeError('historical_prices_tickers.json and datasets_info.json are not consistent')

    def get_time_frame_name(self, ticker, tf='D'):
        """
        Return the correct timeframe name based on the ticker type and data source.

        The methods retrieves the correct time frame name for the price data source from api2source_timeframe_dict,
        which is define in a specific

        Args:
            ticker (str):   name of the ticker
            tf (str):       standard name of the timeframe, from ['M', 'W', 'D', 'H1', 'M30', 'M15', 'M5', 'M1']
                            optional.
                            default: 'D'
        Returns:
            time frame name to use for the data source API, as a str
        """

        instrument_type = self.get_instrument_type(ticker=ticker)
        tf_name = self.api2source_timeframe_dict[tf][instrument_type]['name']
        return tf_name

    def get_instrument_type(self, ticker):
        """
        Returns the type of financial instrument based on the ticker name

        Args:
            ticker (str):   ticker for the instrument

        Returns:
            instrument type such as 'FX', 'STOCK', 'CASH CFD', 'FUTURE CFD' as a str
        """
        list_for_fx = ['Margin FX Contract']
        list_for_stock = ['Stock in major index']
        list_for_cash_cfd = ['Index Cash CFDs', 'Commodity Cash CFDs' ]
        list_for_future_cfd = ['Index Future CFDs', 'Commodity Future CFDs']

        if self.ticker_dictionary[self.source2ticker_name[ticker]]['type'] in list_for_fx:
            instrument_type = 'FX'
        elif self.ticker_dictionary[self.source2ticker_name[ticker]]['type'] in list_for_stock:
            instrument_type = 'STOCK'
        elif self.ticker_dictionary[self.source2ticker_name[ticker]]['type'] in list_for_cash_cfd:
            instrument_type = 'CASH CFD'
        elif self.ticker_dictionary[self.source2ticker_name[ticker]]['type'] in list_for_future_cfd:
            instrument_type = 'FUTURE CFD'
        else:
            instrument_type = 'OTHERS'

        return instrument_type

    def get_price_dict(
        self, 
        ticker_filter:Optional[str]=None,    # Name of the ticker for which to load prices, or wildcard *
        timeframe_filter:Optional[str]=None, # Name of the timeframe considered: FX_INTRADAY_60min/30min/15min/5min/1min FX_DAILY, STOCK_DAILY, FX_WEEKLY, FX_MONTHLY or wildcard *
        time_filter:Optional[str]=None       # Pattern to select a subset of the files based on date: '', 'YYYY', 'YYYY-MM'
        )->dict[str,pd.DataFrame]: # price_dict, dictionary with all uploaded price DataFrames; keys are the filename where the raw price data is stored, values are the pd.DataFrames with the raw price data

        """
        Returns Dict w/ all price files from 'target_directory', filtered by 'ticker' and 'timeframe'.

        Read price data from file for ALL files in 'target_directory' matching passed 'ticker' and 'timeframe'.
        Wildcard * is permitted for both 'ticker' and 'timeframe'.
        Price data follows alphavantage format

        Store prices from each file into a DataFrame
        Clean the data (take out nan)
        Store price DataFrames into a dictionary with key = source price file name

        Note: FOREX and STOCK historical are different: STOCK have a volume columns, FOREX do not
        """

        target_files_list = self.get_raw_price_files(ticker_filter, timeframe_filter, time_filter)

        price_dict = {}
        for file in tqdm(target_files_list):
            prices_df = self.read_csv_price_file(p2file=file)
            prices_df = self.fill_nan(prices_df)
            price_dict[str(file.name)] = prices_df.sort_index(axis='rows', ascending=True)
        return price_dict

    def get_dataset(self, ticker=None, tf=None, adjusted=None, type=None) -> pd.DataFrame:
        """Retrieve the dataset file for passed ticker and timeframe, using current drive (HDD or NAS)

        Args:
            ticker (str):       symbol of ticker for the dataset to retrieve
            tf (str):           one of the timeframe code in self.timeframes
            adjusted (bool):    True: retrieve adjusted stock prices,
                                False: retrieve normal stock prices
                                Optional. Default = True
            type (str):         not implemented yet

        Returns:
            dataset as a Pandas.DataFrame
        """

        tf = self.current_tf = 'D' if tf is None else tf
        ticker = self.current_ticker = 'EURUSD' if ticker is None else ticker
        self.adjusted = True if adjusted is None else adjusted

        if tf not in self.timeframes:
            msg = f"Timeframe <{tf} must be one of these: {self.timeframes}>"
            raise ValueError(msg)
        if ticker not in self.ticker_lists['active']:
            msg = f"Ticker <{ticker} is not in active list for {self.source}>"
            raise ValueError(msg)

        timeframe = self.get_time_frame_name(ticker=ticker, tf=tf)

        # TODO: review this code. Signature of self.get_dataset_filename is not in line with other functions
        instrument_type = self.get_instrument_type(ticker)
        if instrument_type == 'STOCK' and self.adjusted:
            price_type = 'adjusted'
        elif instrument_type == 'STOCK' and not self.adjusted:
            price_type = 'non adjusted'
        else:
            price_type = 'FX'   # FIXME: change this into a bool flag: adj or not adj

        dataset_file_name = self.get_dataset_filename(ticker=ticker, tf=tf, pr_type=price_type)

        path2dataset = self.datasets / dataset_file_name

        if path2dataset.exists():
            logging.info(f'Loading Dataset for {ticker} and {timeframe}')
            ds = pd.read_csv(path2dataset,
                             sep=',',
                             index_col=0,
                             header=None,
                             skiprows=1,
                             parse_dates=[0],
                             names=self.get_dataset_column_names(),
                             na_values=['nan', 'null', 'NULL']
                             )
            return ds
        else:
            logging.warning(f'No dataset file at {path2dataset.absolute()} !!!!!!!!!!!')
            return pd.DataFrame(columns=self.get_dataset_column_names())

    def get_common_dateindex(self, tickers_list, timeframe='D'):
        """
        TODO: docstring
        """
        index_dict = {}
        for ticker in tickers_list:
            ds = self.get_dataset(ticker=ticker, tf=timeframe)
            index_dict[ticker] = set(ds.index)
        common_idx = sorted(list(set.intersection(*list(index_dict.values()))))
        date_start = common_idx[0]
        date_end = common_idx[-1]
        full_time_range = pd.date_range(start=date_start, end=date_end, freq='B')
        missing_dates = set(full_time_range).difference(common_idx)
        return list(common_idx), missing_dates

    def get_prices_from_several_datasets(self, tickers=None, timeframe='D', ohlc='Close', idxs=None):

        if tickers is None:
            tickers = self.ticker_lists['active']
        n = len(tickers)
        assert n >= 2, f"Argument <tickers> must include at least two instruments. {tickers}"

        if idxs is None:
            common_idx, missing_idx = self.get_common_dateindex(tickers_list=tickers, timeframe=timeframe)
        else:
            common_idx = idxs

        # print(f"From {common_idx[0]} to {common_idx[-1]}. {len(common_idx): ,d} bars.")
        df = pd.DataFrame(index=common_idx)
        for ticker in tqdm(tickers):
            ds = self.get_dataset(ticker=ticker, tf=timeframe)
            df[ticker] = ds.loc[common_idx, ohlc]
        return df

    def get_ds_time_range(self, ticker, tf):
        """Returns earliest and latest datetime stored in ds_info time ranges dictionary for the current source"""
        e, l = self.ds_info_dictionary[ticker]['time ranges'].get(self.source, None)[tf]
        e = datetime.strptime(e, '%Y-%m-%d %H:%M:%S') if e is not None else datetime(1, 1, 1, 0, 0, 0)
        l = datetime.strptime(l, '%Y-%m-%d %H:%M:%S') if l is not None else datetime(1, 1, 1, 0, 0, 0)
        return e, l

    @abstractmethod
    def _create_api2source_timeframe_dict(self):
        """
        Template method to return the api2source_timeframe_dict corresponding to the data source.

        Must me implemented in each specific API.
        Methods sets the instance attribute self.api2source_timeframe_dict
        Format for the dictionary:
            keys are the standard timeframes in self.timeframes
            values are sub-dictionaries:
                keys as types of instruments (currently 'FX' and 'STOCK')
                values are sub-dictionaries with at least the key 'name' and others if needed for the API
        """
        pass

    @abstractmethod
    def _define_ticker_lists(self):
        """
        Define standard lists of tickers for the API.

        Must be implemented for each specific API
        Method sets the instance attribute self.ticker_lists, as a dictionary of lists.
        self.ticker_lists is structured as follows:
            keys = names of the list: e.g. active, forex, stocks, index, ...
            values = list of tickers for the instruments in that list
        self.ticker must at least include self.ticker_list['active'] which the list all active instruments
        """
        self.ticker_lists = dict()
        self.ticker_lists['active'] = []
        pass

    @abstractmethod
    def get_raw_price_files(self, ticker_filter, timeframe_filter, time_filter)->list:
        """Gets the list of filenames for all raw historical files matching filters, and return as list

        This methods must be implemented in each specific API
        """
        pass

    @abstractmethod
    def date_latest_file_download(self):
        """Return the date of the last time pricing info was uploaded
        
        This methods must be implemented in each specific API
        """
        pass

    @abstractmethod
    def read_csv_price_file(self, p2file:Path)->pd.DataFrame:
        """Reads the price data from the file at p2file and returns a DataFrame
        
        This methods must be implemented in each specific API
        """
        pass

    @abstractmethod
    def get_dataset_filename(self, ticker:str, tf:str, pr_type:str) -> str:
        """Return the name of the dataset file for the passed ticker, timeframe and price type
        
        This methods must be implemented in each specific API
        """
        pass

    @abstractmethod
    def get_dataset_column_names(self) -> list:
        """Return the list of column names for the dataset file
        
        This methods must be implemented in each specific API
        """
        pass
    
    @staticmethod
    def load_file_from_url(target_url, filepath):
        """
        Download file from passed 'target_url' using request.urlopen method. Save file as filename

        Args:
            target_url (str):   format is http://... or https://....
            filepath (Path):    Path to the file

        Returns:
            True if operation did not raise any issue.
            False otherwise.
        """

        def try_internet_access(function_to_apply, *args, **kwargs):
            """Wrapper function to handle exceptions when accessing URL
            TODO: replace this wrapper with a decorator ?

            Will catch any exception raised while connecting to the server.
            If an exception is caught, generates a message to log and returns False to indicate a failed attempt.
            If no exceptions are caught, returns the result of the function to apply

            Technical references:
                - https://docs.python.org/3.6/library/urllib.request.html
                - also could use http.client library: conn = http.client.HTTPConnection() and conn.request('GET', '/')
                - https://docs.python.org/3.6/library/http.client.html#http.client.HTTPResponse
            """
            try:
                results = function_to_apply(*args, **kwargs)
            except requests.HTTPError as err:
                # https://docs.python-requests.org/en/master/user/quickstart/#errors-and-exceptions
                # https://docs.python-requests.org/en/latest/api/
                logging.warning(f'  >> HTTPError: {err}. URL: {target_url}.')
                return False
            except requests.TooManyRedirects as err:
                # https://docs.python-requests.org/en/master/user/quickstart/#errors-and-exceptions
                # https://docs.python-requests.org/en/latest/api/
                logging.warning(f'  >> TooManyRedirects: {err}. URL: {target_url}.')
                return False
            except requests.ConnectTimeout as err:
                # https://docs.python-requests.org/en/master/user/quickstart/#errors-and-exceptions
                # https://docs.python-requests.org/en/latest/api/
                logging.warning(f'  >> ConnectTimeout: {err}. URL: {target_url}.')
                return False
            except requests.ReadTimeout as err:
                # https://docs.python-requests.org/en/master/user/quickstart/#errors-and-exceptions
                # https://docs.python-requests.org/en/latest/api/
                logging.warning(f'  >> ReadTimeout: {err}. URL: {target_url}.')
                return False
            except requests.Timeout as err:
                # https://docs.python-requests.org/en/master/user/quickstart/#errors-and-exceptions
                # https://docs.python-requests.org/en/latest/api/
                logging.warning(f'  >> Timeout: {err}. URL: {target_url}.')
                return False
            except requests.ConnectionError as err:
                # https://docs.python-requests.org/en/master/user/quickstart/#errors-and-exceptions
                # https://docs.python-requests.org/en/latest/api/
                logging.warning(f'  >> ConnectionError: {err}. URL: {target_url}.')
                return False
            except requests.RequestException as err:
                # https://docs.python-requests.org/en/master/user/quickstart/#errors-and-exceptions
                # https://docs.python-requests.org/en/latest/api/
                logging.warning(f'  >> RequestException: {err}. URL: {target_url}.')
                return False
            except ssl.SSLError as err:
                # https://docs.python.org/3.6/library/ssl.html
                # https://docs.python.org/3.6/library/ssl.html#ssl.SSLError
                logging.warning(f"  >> ssl.SSLError: {err}")
                return False
            except socket.timeout as err:
                # https://stackoverflow.com/questions/45774126/catching-socket-timeout-the-read-operation-timed-out-in-python
                # https://docs.python.org/3.6/library/socket.html#exceptions
                logging.warning(f"  >> socket.timeout error: {err}")
                return False
            except http.client.HTTPException as err:
                # https://docs.python.org/3.6/library/http.client.html#http.client.HTTPException
                logging.warning(f"  http.client.HTTPException: {err}>> ")
                return False
            else:
                return results

        logging.info(f'  - Contacting server at {target_url}')
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36",
            "Accept-Encoding": "*"
        }
        # try: response = requests.get() through the wrapper
        response = try_internet_access(requests.get, target_url, headers=headers, timeout=60)

        if response is False:
            return False
        else:
            with open(filepath, 'wb') as f:
                f.write(response.content)
                logging.info(f'  - Done writing data into file {filepath}')
            return True

    @staticmethod
    def exclude_idx_from_df(
        df_with_idx_to_exclude:pd.DataFrame, # the DataFrame whose index will be excluded from df_to_filter
        df_to_filter: pd.DataFrame  # the DataFrame that will be filtered.
        ) -> pd.DataFrame:  # filtered DataFrame
        """
        Excludes rows in df_to_filter with index values also in the index of df_with_idx_to_exclude.

        Extracts both DataFrames index, transforms them into sets and filters df_to_filter with the df2_index
        excl. df1.index
        """
        idx_to_exclude = df_with_idx_to_exclude.index
        idx_to_filter = df_to_filter.index
        filtered_idx = list(set(idx_to_filter).difference(set(idx_to_exclude)))
        return df_to_filter.loc[filtered_idx, :]

    @staticmethod
    def remove_multiple_wildcards(s):
        """
        Return a str where any occurrence of multiple * is replace by a single *.

        FIXME: this only works for up to 10 wildcards. For more, need to use regex

        Args:
            s (str):    String from which multiple wildcards must be removed

        Returns:
            Clean string with single wildcards
        """
        return s.replace('***', '*').replace('**', '*').replace('**','*')

    @staticmethod
    def fill_nan(df):
        """
        Fill DataFrame nan in two passes: forward fill first, then back fill nan at start of column

        Args:
            df (pd.DataFrame):  DataFrame from which the NaNs must be filled

        Returns:
            pd.DataFrame with all NaNs filled
        """
        df_filled = df.fillna(method='ffill', inplace=False)
        df_filled.fillna(method='bfill', inplace=True)
        return df_filled

    @staticmethod
    def split_text_in_lines(txt, line_max_length=80):
        """Split a text in several lines based on the max line length passed

        Args:
            txt (str):              the text to split into lines
            line_max_length (int):  maximum character length of a line
        Returns:
            The list of lines (list of str)

        """
        list_of_words = txt.split()
        list_of_lines = []

        line = ''
        while len(list_of_words) > 0:
            w = list_of_words.pop(0)
            if len(line) + len(w) > line_max_length:
                list_of_lines.append(line)
                line = ''
            line = line + w + ' '
        list_of_lines.append(line)

        return list_of_lines


class AlphavantageAPI(API):
    """API for Alphavantage


    Attributes:
        self.source (str):       name of the data source for this API. Shared by all instances
        self.data_format (str)   name of the data format for this API. Shared by all instances

    """

    source = 'alphavantage'
    folder = 'alphavantage'
    data_format = 'alphavantage'

    def __init__(self):
        """Specific initialisation for Alphavantage

        Attributes:
            self.api_key_ring (ApiKeyRing):
            self.api2source_timeframe_dict:
            self.api_info_dict:
            self.source2ticker_name:
            self.current_ticker:
            self.current_tf:
            self.throttle_delay (int):
            self.ticker_lists:
            self.methods:
            self.adjusted:
            self.mode:

        """
        super().__init__()
        self.api_key_ring = ApiKeyRing(config=self.config, source=self.source)
        # self.throttle_delay = 2 + 20 / len(self.api_key_ring)  # Max API frequency is 5 calls/min and 500 calls/day
        self.throttle_delay = 20  # seems that Alphavantage identifies IP address, cannot trick with several keys

        self.methods = [self.info, self.info_methods,
                        self.retrieve_prices, self.search_for_stock_info,
                        self.get_dataset, self.update_price_datasets,
                        self.get_common_dateindex,
                        self.date_latest_file_download,
                        self.get_prices_from_several_datasets,
                        self.switch_to_HDD, self.switch_to_NAS]

        self.current_ticker:str = ''
        self.current_tf:str = ''
        self.adjusted = True
        self.mode = 'compact'

        logging.info(f"API object created, with {len(self.ticker_lists)} predefined lists of tickers "\
                     f"and {len(self.methods)} API methods")
        logging.info(f"ticker lists:\n{[key for key in self.ticker_lists.keys()]}")
        logging.info(f"methods:\n{[fct.__name__ for fct in self.methods]}")

    def _create_api2source_timeframe_dict(self):
        """
        Creates the api2source_timeframe_dict attribute for AlphavantageAPI

        Load alphavantage timeframe dictionary from json file and creates the attribute  api2source_timeframe_dict

        Technical Note:
            uses pkg_resources to point to package directory

        JSON format:
            "M1": {
                    "mode": "compact",
                    "FX": {
                            "name": "FX_INTRADAY_1min",
                            "url-param": "FX_INTRADAY&interval=1min"
                            },
                    "STOCK": {
                             "name": "STOCK_INTRADAY_1min",
                             "url-param": "TIME_SERIES_INTRADAY&interval=1min"
                  }

        """

        p2json_file = PACKAGE_ROOT / 'assets/json/alphavantage-timeframes.json'
        if p2json_file.is_file():
            with open(p2json_file, 'r') as fp:
                self.api2source_timeframe_dict = json.load(fp)
        else:
            raise ValueError(f"No json file with name <{p2json_file.name}> at {p2json_file.absolute()}")

    def _define_ticker_lists(self):
        """
        Define standard lists of tickers for AlphavantageAPI
        """
        self.ticker_lists = dict()
        self.ticker_lists['forex'] = sorted([subdict['tickers'][self.source]
                                             for _, subdict in self.ticker_dictionary.items()
                                             if self.source in subdict['tickers'].keys()
                                             and subdict['type'] == 'Margin FX Contract'])
        self.ticker_lists['stock'] = sorted([subdict['tickers'][self.source]
                                             for _, subdict in self.ticker_dictionary.items()
                                             if self.source in subdict['tickers'].keys()
                                             and subdict['type'] == 'Stock in major index'])
        self.ticker_lists['active'] = self.ticker_lists['forex'] + self.ticker_lists['stock']

    def build_url(self):
        """Build URL for alphavantage API, based on ticker, tf and other relevant attributes saved in API.

        Examples of Alphavantage URLs
        =============================
        Forex
            intraday:    https://www.alphavantage.co/query?function=FX_INTRADAY&interval=5min&from_symbol=EUR&to_symbol=USD&outputsize=compact&apikey=demo&datatype=csv
            daily:       https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=EUR&to_symbol=USD&outputsize=compact&apikey=demo&datatype=csv
            weekly:      https://www.alphavantage.co/query?function=FX_WEEKLY&from_symbol=EUR&to_symbol=USD&outputsize=compact&apikey=demo&datatype=csv
        Stock
            intraday:   https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&interval=5min&symbol=MSFT&outputsize=compact&apikey=demo&datatype=csv
            daily:      https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&outputsize=compact&apikey=demo&datatype=csv
            weekly:     https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=MSFT&outputsize=compact&apikey=demo&datatype=csv
        Stock adjusted:
            daily:      https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=IBM&apikey=demo&datatype=csv
            weekly:     https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol=IBM&apikey=demo

        API Documentation
        =================
        https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=AUD&to_symbol=JPY&outputsize=compact&apikey=EBPY7H5F030LKZJM&datatype=csv

        Args:
            The method uses values of several instance attributes and there is no need to pass any argument.
            Used instance attributes: self.current_ticker, self.current_tf, self.mode and self.adjusted

        Returns:
            url to use to retrieve the file from Alphavantage API, as a str
        """

        ticker_name, tf = self.current_ticker, self.current_tf

        if self.ticker_dictionary[self.source2ticker_name[ticker_name]]['type'] == 'Margin FX Contract':
            function_str = self.api2source_timeframe_dict[tf]['FX']['url-param']
            target_currency = ticker_name[0:3]
            base_currency = ticker_name[3:6]
            symbol_str = f"&from_symbol={target_currency}&to_symbol={base_currency}"
        else:
            function_str = self.api2source_timeframe_dict[tf]['STOCK']['url-param'] + self.get_adjusted_string()
            symbol_str = f"&symbol={ticker_name}"

        api_key = self.api_key_ring.get_new_key()

        url = 'https://www.alphavantage.co/query?function=' + function_str + symbol_str \
              + '&outputsize=' + self.mode \
              + '&apikey=' + api_key \
              + '&datatype=csv'

        return url

    def get_price_file_correct_header_line(self):
        """
        Retrieves historical price file headers based on the ticker saved in instance attribute: `current_ticker`.

        Implementation for Alphavantage API.
        If the the symbol in self.current_ticker is not in the active list, raises ValueError

        Returns:
            list of the header names in the historical price file
        """

        ticker = self.current_ticker
        if ticker in self.ticker_lists['active']:
            instrument_type = self.get_instrument_type(ticker)
            if instrument_type == 'FX':
                return 'timestamp,open,high,low,close'
            elif instrument_type == 'STOCK':
                if self.adjusted:
                    if self.current_tf in ['M', 'W']:
                        return 'timestamp,open,high,low,close,adjusted close,volume,dividend amount'
                    elif self.current_tf in ['D']:
                        return 'timestamp,open,high,low,close,adjusted_close,volume,dividend_amount,split_coefficient'
                    else:
                        return 'timestamp,open,high,low,close,volume'
                else:
                    return 'timestamp,open,high,low,close,volume'
            else:
                raise ValueError(f"{instrument_type} is not a known instrument type for {self.source}")

        else:
            # Not a ticker (e.g. a key word used for search API) and no header line should be returned.
            return None

    def get_adjusted_string(self):
        """Return '_ADJUSTED' for adjusted stock prices and time frames 'D', 'W', 'M'

        Returns:
            str to add for adjusted stock prices

        TODO: Refactor or delete after making all stock retrieval adjusted
        """
        if self.get_instrument_type(self.current_ticker) in ['FX']:
            return ''
        elif self.get_instrument_type(self.current_ticker) in ['STOCK']:
            return '_ADJUSTED' if self.adjusted and self.current_tf in ['D', 'W', 'M'] else ''

    def handle_file_error_message(self, filepath):
        """Check that downloaded file is valid. Otherwise, handle API quota errors

        E.g. proper headers must be: 'timestamp,open,high,low,close,volume'
            "Error Message": "Invalid API call. Please retry or visit the documentation

        Args:
            filepath (Path):    path to the file to test

        Returns:
            True is there is no error message in the downloaded file,
            False if there is an error
        """
        file_loaded_correctly, api_quota_error = False, False

        # verify that the file header is correct
        correct_header = self.get_price_file_correct_header_line()
        if correct_header is not None:
            with open(filepath, 'rt') as f:
                first_line = f.readline()
            file_loaded_correctly = correct_header in first_line

        # no standard header, check whether there is any error message pattern
        else:
            with open(filepath, 'rt') as f:
                file_content = " ".join([line for line in f.readlines(1024)])
            error_1_pattern = 'Error Message'
            error_2_pattern = '{}'
            error_1 = error_1_pattern in file_content
            error_2 = error_2_pattern in file_content
            file_loaded_correctly = not any([error_1, error_2])

        # if there is a problem with the file, check that whether it is API quota related
        if not file_loaded_correctly:
            with open(filepath, 'rt') as f:
                file_content = " ".join([line for line in f.readlines(1024)])

            exceeds_api_quota = 'Our standard API call frequency'
            api_quota_error = exceeds_api_quota in file_content
            if api_quota_error:
                self.handle_api_error()

            logging.warning(f"  - !!!! {self.current_ticker}-{self.current_tf}. Error message in the downloaded file:")
            for line in self.split_text_in_lines(txt=file_content, line_max_length=80):
                logging.info(f"           {line}")

        # Returns True if the file does not include error message. Also returns api_quota_error, True if api error
        return file_loaded_correctly, api_quota_error

    def handle_api_error(self):
        """Handles keys when an API quota error is detected

        If there are enough keys, removes the key under which the quota error was raised.
        Throttle delay is also adjusted to match the number of keys.
        If there are not enough keys, reinitialize the full set of keys and pauses for 5 min.
        """
        last_used_api_key = self.api_key_ring.read_key()
        msg = f"  - !!!! {self.current_ticker}-{self.current_tf}. API usage error with " \
              f"{last_used_api_key[0:5]}{'*'*11}. Quota exceeded."
        logging.warning(msg)

        if len(self.api_key_ring) > 1:
            # Remove current API key for which the quota was exceeded
            removed_api_key = self.api_key_ring.popleft()
            msg = f"  - !!!! {len(self.api_key_ring)} keys remaining, removing {removed_api_key[0:5]}{'*'*11} from ring"
            logging.warning(msg)

            # Next time a new key will be requested, the ring will be rotated one step, therefore we need to rotate it
            # in the other direction to position the intended next key as second to left
            #  Ring: |1|2|3|4|5| before removing the error key. Next new key is |2|
            #  Ring: |2|3|4|5|   after removing the error key. If we apply .get_new_key(),
            #                    it will return |3| and the ring will have rotated as  |3|4|5|2|, we want |2|3|4|5|
            most_right_api_key = self.api_key_ring.pop()
            self.api_key_ring.appendleft(most_right_api_key)

            # update throttle delay to the actual number of API keys
            self.throttle_delay = 2 + 20 // len(self.api_key_ring)

        else:
            logging.warning(f"  - !!!! All key over quota. Resetting all keys and throttling for 5 min")
            self.api_key_ring = ApiKeyRing(config=self.config, source=self.source)
            time.sleep(5 * 60)
            logging.info('  - Awaking from 5 min throttle phase\n')

    def throttle_api(self):
        """Execute the throttle actions. Current implementation is a simple sleep time."""
        logging.info(f'  - Throttle API for {self.throttle_delay} sec.')
        time.sleep(self.throttle_delay)
        logging.info(f'  - Awaking from {self.throttle_delay} sec throttle phase\n')

    def retrieve_prices(self, mode='compact', tickers=None, timeframes=None, skip_existing=False,
                        adjusted=True):
        """Get instrument pricing data from alphavantage for default tickers and tf or specified ones.

        Note:
            From 2020-08-18 this will only retrieve stock prices in Adjusted Price format.
            Retrieving non adjusted price format is deprecated

        Args:
            mode (str):             'full' or 'compact'
                                    default = 'compact
            tickers (list(str)):    such as ['EURUSD', 'USDJPY', 'MSFT', 'RSA.LON']
                                    optional
                                    default is full active list of tickers
            timeframes (list(str)): subset of ['M', 'W', 'D', 'H1', 'M30', 'M15', 'M5', 'M1']
                                    default is ['D', 'H1', 'M30']
            skip_existing (bool):   If True, skip the download of a file in case of pre-existing file with the
                                    same name. Used to speed up re-updated of historical data when done the same day.
                                    optional
                                    default is False
            adjusted (bool):        Deprecated argument since Aug 2020. Must be True if passed, no other value accepted
                                    default is True

        """

        # Setup infra and parameters for class
        top_box = InfoBox()

        msg = f'Starting alphavantage price update'
        logging.info(msg)
        top_box.print(msg)

        if adjusted is False:
            msg = f"adjusted is a deprecated argument since Aug 2020. " \
                  f"Only 'True' is a valid value if the argument is passed, not {adjusted}."
            raise ValueError(msg)

        self.adjusted = True
        self.mode = mode

        if tickers is None:
            tickers = sorted(list(set(self.ticker_lists['tickers_active'])))
        if timeframes is None:
            timeframes = ['D', 'H1', 'M30']

        number_tickers = len(tickers)
        number_timeframes = len(timeframes)

        self.downloads_to_retry_dict = {}
        self.failed_downloads_dict = {}

        logging.info(f"Get new data for selected tickers ({number_tickers}) and timeframes ({number_timeframes}):")
        logging.info(f'  - {timeframes}')
        logging.info(f'  - {tickers}')

        # Opening message on screen
        api_calls_stocks = number_tickers * number_timeframes
        print('Summary of API access parameters:')
        print(f"- Run on {datetime.now().strftime('%b %d, %Y %H:%M')} local time")
        print(f"- Total Nbr API calls: {api_calls_stocks}:")
        print(f"   - {number_tickers:,d} tickers")
        print(f"   - {number_timeframes:,d} time frames: {timeframes}")

        tf_bar = tqdm(timeframes)
        for i, tf in enumerate(tf_bar, 1):
            self.current_tf = tf
            now = datetime.today()
            now_string = str(now.year) + '-' + str(now.month).zfill(2) + '-' + str(now.day).zfill(2)
            if mode == 'full':
                mode = mode
            else:
                mode = self.api2source_timeframe_dict[tf]['mode']
                # Overwrites mode from compact to full for shortest time frame (as per 'alphavantage-timeframes.json')

            string_to_print = f"\nDownloading all tickers for {tf} (mode={mode})"
            tf_bar.set_description(f"Downloading all tickers for {tf}")
            tf_bar.set_postfix(mode=mode)
            logging.info(string_to_print)

            ticker_bar = tqdm(tickers)
            for counter, ticker in enumerate(ticker_bar, 1):
                self.current_ticker = ticker

                string_to_print = f"    {ticker}"
                ticker_bar.set_description(string_to_print)
                logging.info(string_to_print)

                # TODO: replace ticker.replace() by a method usable across the classes
                filename = f"{ticker.replace('.', '-')}_{self.get_time_frame_name(ticker, tf)}_{now_string}.csv"
                filepath = self.raw_data / filename
                if skip_existing and filepath.exists():
                    logging.info(f"  - {filepath.name} already exists, skipping this download")
                    continue

                url = self.build_url()
                load_from_url_succeeded = self.load_file_from_url(target_url=url, filepath=filepath)

                if load_from_url_succeeded is True:
                    logging.info(f"  - Done writing data into {filename}.")
                    file_downloaded_correctly, api_usage_error = self.handle_file_error_message(filepath=filepath)

                    if file_downloaded_correctly is False:
                        logging.warning(f"  - !!!! {self.current_ticker}-{self.current_tf}. Data in file not correct.")
                        path_file_to_download = str(filepath.absolute())
                        self.downloads_to_retry_dict[path_file_to_download] = {'url': url,
                                                                               'ticker': ticker,
                                                                               'tf': tf,
                                                                               'mode': mode,
                                                                               'count': 0,
                                                                               }
                        if filepath.is_file():
                            os.remove(filepath)
                        logging.info(f"         Ticker added to retry list. Deleted erroneous file.")

                else:
                    logging.warning(f"  - !!!! {self.current_ticker}-{self.current_tf}. Failed GET to {url}.")
                    path_file_to_download = str(filepath.absolute())
                    self.downloads_to_retry_dict[path_file_to_download] = {'url': url,
                                                                           'ticker': ticker,
                                                                           'tf': tf,
                                                                           'mode': mode,
                                                                           'count': 0,
                                                                           }
                    logging.info(f"         Ticker added to retry list.")

                self.throttle_api()

        # One automatic retry for failed downloads. If fails again, give instructions for manual retry
        if self.downloads_to_retry_dict:
            string_to_print = f'Starting retry for failed ticker downloads'
            logging.info(string_to_print)
            top_box.print(string_to_print)

            self.instructions_list = []
            for path, values in self.downloads_to_retry_dict.items():

                self.current_ticker, self.current_tf = values['ticker'], values['tf']
                failed_a_second_time = False

                string_to_print = f'\n  Retry for {values["ticker"]}'
                logging.info(string_to_print)
                # low_box.print(string_to_print)

                filepath = Path(path)
                load_from_url_succeeded = self.load_file_from_url(target_url=values['url'], filepath=filepath)

                if load_from_url_succeeded is False:
                    failed_a_second_time = True
                else:
                    file_downloaded_correctly, api_usage_error = self.handle_file_error_message(filepath=filepath)

                    if file_downloaded_correctly is False:
                        failed_a_second_time = True

                if failed_a_second_time:
                    if filepath.is_file():
                        os.remove(filepath)

                    self.failed_downloads_dict[path] = {'ticker': values['ticker'],
                                                        'tf': values['tf'],
                                                        'mode': values['mode'],
                                                        }

                    logging.warning(f"  - !!!! {values['ticker']}-{values['tf']}. File incorrect a second time.")
                    logging.warning(f"  - !!!! To retry manually, instructions listed at the end of this log")

                    function_name = 'av.retrieve_prices'
                    parameters_1 = f'tickers=["{values["ticker"]}"], timeframes=["{values["tf"]}"], '
                    parameters_2 = f'mode="{values["mode"]}", skip_existing=False'
                    instruction = function_name + '(' + parameters_1 + parameters_2 + ')'
                    self.instructions_list = self.instructions_list + [instruction]

                self.throttle_api()

            string_to_print = 'Automatic retry completed'
            logging.info(string_to_print)
            top_box.print(string_to_print)

            if self.instructions_list:
                print('\nInstructions to manually retry failed downloads:')
                print('------------------- Instruction Section Start -----------------------------')
                for instruction in self.instructions_list:
                    print(instruction)
                print('--------------------- Instruction Section End -----------------------------')

        string_to_print = f'\nCompleted alphavantage price update\n'
        top_box.clear()
        # low_box.clear()
        # low_box.print(string_to_print)
        logging.info(string_to_print)

        print(f"- Run ended on {datetime.now().strftime('%b %d, %Y %H:%M')} local time")

        return None

    def retrieve_company_info(self, tickers=None):
        """
        TODO: create the code to retrieve company information and statement

        URL for Company summary info
        https://www.alphavantage.co/query?function=OVERVIEW&symbol=IBM&apikey=demo
        URL for income statements
        https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=IBM&apikey=demo
        URL for balance sheet statements
        https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol=IBM&apikey=demo
        URL for cash flow statements
        https://www.alphavantage.co/query?function=CASH_FLOW&symbol=IBM&apikey=demo

        Args:
            tickers  (list(str)):   list with all tickers for companies to retrieve info from
        """

        raise NotImplementedError('This method is not implemented yet')

    def search_for_stock_info(self, search_words=''):
        """Access the 'Search Endpoint' API by Alphavantage. Retrieves symbols and companies from search words

        Args:
        search_words (list(str)):   Keywords to use for the search.
                                    Keyword may not include white spaces
        """

        downloads_to_retry_dict = {}
        number_search_words = len(search_words)
        search_results = ''

        logging.info(f'Get company info for selected search words ({number_search_words}):')
        logging.info(f'  - {search_words}')

        for counter, sw in enumerate(tqdm(search_words), 1):
            string_to_print = f"\n  Starting process for {sw} (search word {counter} of {number_search_words})"
            logging.info(string_to_print)

            filename = 'search_result.csv'
            filepath = (PACKAGE_ROOT / f"../data/raw-data/alphavantage/{filename}").resolve()

            api_key = self.api_key_ring.get_new_key()
            url_seed = 'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&datatype=csv'
            url_search = f"&keywords={sw}"
            url_apikey = f"&apikey={api_key}"
            url = url_seed + url_search + url_apikey

            load_from_url_succeeded = self.load_file_from_url(target_url=url, filepath=filepath)
            if load_from_url_succeeded is True:
                logging.info(f'  - Done writing data into {filename}.')

                _, api_quota_error = self.handle_file_error_message(filepath=filepath)

                if api_quota_error:
                    path_file_to_download = str(filepath.absolute())
                    downloads_to_retry_dict[path_file_to_download] = {'url': url,
                                                                      'ticker': sw,
                                                                      'tf': None,
                                                                      'mode': None,
                                                                      'count': 0,
                                                                      }

                else:
                    results_columns = ['symbol', 'name', 'type', 'region', 'marketOpen',
                                       'marketClose', 'timezone', 'currency', 'matchScore']
                    sr_ds = pd.read_csv(filepath,
                                        sep=',',
                                        index_col=0,
                                        header=None,
                                        skiprows=1,
                                        names=results_columns,
                                        na_values=['nan', 'null', 'NULL']
                                        )
                    print(f"Results for search word: {sw}:")
                    display(sr_ds)

                os.remove(filepath)
                self.throttle_api()

    def get_dataset_column_names(self):
        """Return the name of dataset columns depending on the ticker stored in instance attribute current_ticker

        Currently implemented for FX and STOCK.
        Currently keep the option for attribute adjusted to be false, in order to load existing non adjusted datasets

        Returns:
            List of columns name/headers for the dataset
        """
        ticker, tf = self.current_ticker, self.current_tf

        instrument_type = self.get_instrument_type(ticker)
        if instrument_type == 'FX':
            return ['timestamp', 'Open', 'High', 'Low', 'Close']
        elif instrument_type == 'STOCK':
            if self.adjusted:
                if self.current_tf in ['M', 'W']:
                    return ['timestamp', 'Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume',
                            'Dividend_amount']
                elif self.current_tf in ['D']:
                    return ['timestamp', 'Open', 'High', 'Low', 'Close', 'Adjusted_Close', 'Volume',
                            'Dividend_amount', 'Split Coefficient']
                else:
                    return ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            else:
                return ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        else:
            raise ValueError(f"{instrument_type} is not a known instrument type for {self.source}")
            return False

    def get_dataset_filename(self, ticker:str, tf:str, pr_type=None) -> str:
        """
        Returns the file name for a dataset according to the ticket and timeframe

        Args:
            ticker (str):   Ticker for the dataset to retrieve
            tf (str):       TimeFrame
            pr_type (str):  Value in ['adjusted', 'non adjusted']
                            Optional. Default value: 'adjusted'
        Returns:
            Filename formed from the passed arguments, as a str

        """

        pr_type = 'adjusted' if pr_type is None else pr_type

        if self.get_instrument_type(ticker) == 'STOCK' and pr_type == 'non adjusted':
            return f"{self.source}-{ticker.replace('.', '-')}-{self.get_time_frame_name(ticker, tf)}-non-adjusted.csv"
        else:
            return f"{self.source}-{ticker.replace('.', '-')}-{self.get_time_frame_name(ticker, tf)}.csv"

    def get_raw_price_files(self, ticker_filter, timeframe_filter, time_filter):
        """
        Retrieve the raw price files from the raw_data folder, possibly filtered

        Returns:
            The list with all files Paths
        """

        ticker_filter = '*' if ticker_filter is None else ticker_filter
        timeframe_filter = '*' if timeframe_filter is None else timeframe_filter
        time_filter = '' if time_filter is None else time_filter

        raw_selection_pattern = f"{ticker_filter.replace('.', '-')}_{timeframe_filter}_{time_filter}*.csv"
        selection_pattern = self.remove_multiple_wildcards(raw_selection_pattern)
        target_files_list = list(self.raw_data.glob(selection_pattern))
        return target_files_list

    def read_csv_price_file(self, p2file):
        """
        Read historical price csv file into a Dataframe, according the the format corresponding to the API: Alphavantage

        Args:
            p2file (Path):  path the the file to read into DataFrame

        Returns:
            DataFrame with the prices
        """
        df = pd.read_csv(p2file,
                         sep=',',
                         index_col=0,
                         header=None,
                         skiprows=1,
                         parse_dates=True,
                         names=self.get_dataset_column_names(),
                         na_values=['nan', 'null', 'NULL']
                         )
        return df

    def update_price_datasets(
        self, ticker_list:Optional[list[str]]=None, 
        timeframe_list:Optional[list[str]]=None, 
        timefilter:Optional[str]=None, # used to filter the files to use for updating the dataset.
        price_type:Optional[str]=None  # deprecated. forced to 'adjusted' for backwards compatibility
        ) -> None:
        """
        Update all dataset for the tickers in the ticker list and timeframe list
        By default will update both adjusted and non adjusted prices datasets.

        Note:
            Since Aug 2020, datasets for stocks are only updated with adjusted prices. Type "non-adjusted" is deprecated
        """
        top_box = InfoBox()
        low_box = InfoBox()

        if timeframe_list is None:
            timeframe_list = ['D', 'H1', 'M30']
        if ticker_list is None:
            ticker_list = self.ticker_lists['forex']
        if price_type is None:
            price_type = 'adjusted'
        elif price_type in ['both', 'non-adjusted']:
            msg = f"price_type is a deprecated argument. " \
                  f"Only value accepted for backward compatibility is 'adjusted' and not '{price_type}'"
            raise ValueError(msg)
        if timefilter is None:
            timefilter = '2'    # ensure that the sequence is a datetime in format 2yyy-mm-dd

        for ticker in tqdm(ticker_list):
            self.current_ticker = ticker
            for timeframe in tqdm(timeframe_list):
                self.current_tf = timeframe

                logging.info(f'Handling {ticker} for {timeframe}')
                dataset_file_name = self.get_dataset_filename(ticker, timeframe)
                path2dataset = self.datasets / dataset_file_name

                price_dict = self.get_price_dict(ticker_filter=ticker,
                                                 timeframe_filter=self.get_time_frame_name(ticker, timeframe),
                                                 time_filter=timefilter)
                list_of_files = list(price_dict)

                if len(list_of_files) == 0:
                    logging.warning(f"No {timeframe} prices for {ticker}. Skipping to next")
                    break

                if path2dataset.exists():
                    logging.info(f' - Loading existing dataset from {dataset_file_name}')
                    full_ds = self.get_dataset(ticker=ticker, tf=timeframe)

                else:
                    logging.info(f' - Creating a new dataset for {dataset_file_name}')
                    # columns do not include the timestamp, which is the name of the index
                    full_ds = pd.DataFrame(columns=self.get_dataset_column_names()[1:])

                for file_name, df in price_dict.items():
                    # We do not use the last bar of df because it may not be fully formed, hence df.iloc[:-1, :]
                    filtered_df = self.exclude_idx_from_df(
                        df_with_idx_to_exclude=full_ds,
                        df_to_filter=df.iloc[:-1, :]
                        )
                    logging.info(f' - Updating with {file_name}')
                    logging.info(
                       f'   Dataset: {full_ds.shape[0]}. File: {df.shape[0]}. Added {filtered_df.shape[0]} rows.')

                    full_ds = pd.concat([full_ds,filtered_df], verify_integrity=True, sort=False)
                    full_ds.sort_index(ascending=True, inplace=True)

                assert all(full_ds.isna()) is True, f'Some values in the dataset (full_ds) are NaN.'

                logging.info(f' - Saving updated dataset into {dataset_file_name}')
                full_ds.to_csv(path_or_buf=self.datasets / dataset_file_name, index_label='timestamp')

                # Update the time range info in ds_info_dictionary if full_ds is not empty
                if full_ds.shape[0] > 0:
                    logging.info(f" - Updating the dataset info dictionary with new time range")
                    time_range = (full_ds.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                                  full_ds.index[-1].strftime('%Y-%m-%d %H:%M:%S'))
                    self.ds_info_dictionary[self.source2ticker_name[ticker]]['time ranges'][self.source][timeframe] = time_range

        # Update datasets_info.json based updated ds_info_dictionary (use pkg_resources to point to package directory)
        # TODO: add a handle_json(json_file_name, action) method for this code that is repeated several times
        p2json_file = Path(pkg_resources.resource_filename(__name__, 'datasets_info.json'))
        if p2json_file.is_file():
            with open(p2json_file, 'w') as fp:
                json.dump(self.ds_info_dictionary, fp, indent=4)
        else:
            raise ValueError(f"No json file with name <{p2json_file.name}> at {p2json_file.absolute()}")

    def date_latest_file_download(self):
        """Return the date of the last time pricing info was uploaded"""
        list_of_file_names_fx = list(p.name for p in self.raw_data.glob('EURUSD_FX_DAILY_*'))
        list_of_file_names_stock_adj = list(p.name for p in self.raw_data.glob('A_STOCK_DAILY_ADJUSTED*'))
        list_of_file_names_stock = list(p.name for p in self.raw_data.glob('A_STOCK_DAILY_*'))
        list_of_file_names_stock = list(set(list_of_file_names_stock).difference(set(list_of_file_names_stock_adj)))

        # EURUSD_FX_DAILY_YYYY-MM-DD.csv 0123456789012345 YYYY 0 MM 3 DD 4
        list_of_dates_fx = [datetime(year=int(s[16:20]),
                                     month=int(s[21:23]),
                                     day=int(s[24:26])) for s in list_of_file_names_fx]

        # A_STOCK_DAILY_YYYY-MM-DD.csv   01234567890123 YYYY 8 MM 1 DD 4567
        list_of_dates_stock = [datetime(year=int(s[14:18]),
                                        month=int(s[19:21]),
                                        day=int(s[22:24])) for s in list_of_file_names_stock]
        #                        2  2 2 33 3
        #                        3  6 8 01 3
        # A_STOCK_DAILY_ADJUSTED_YYYY-MM-DD.csv   01234567890123 YYYY 8 MM 1 DD 4567
        list_of_dates_stock_adj = [datetime(year=int(s[23:27]),
                                            month=int(s[28:30]),
                                            day=int(s[31:33])) for s in list_of_file_names_stock_adj]

        list_all_dates = list_of_dates_fx + list_of_dates_stock + list_of_dates_stock_adj

        if list_all_dates:
            return max(list_all_dates)
        else:
            print('No historical price file in the folder')
            return datetime(year=1900, month=1, day=1)


class AxitraderAPI(API):
    """
    API for Axitrader MT4

    Attributes:
        self.source (str):       name of the data source for this API. Shared by all instances
        self.data_format (str)   name of the data format for this API. Shared by all instances
    """

    source = 'axitrader'
    folder = 'axitrader-mt4'
    data_format = 'mt4'

    def __init__(self):
        """
        Specific initialisation for Axitrader MT4

        Args:
            self.api2source_timeframe_dict:
            self.source2ticker_name:
            self.ticker_lists:

        """
        super().__init__()
        self.methods = [self.info, self.info_methods,
                        self.get_dataset, self.update_price_datasets,
                        self.date_latest_file_download,
                        self.switch_to_HDD, self.switch_to_NAS]

        self.current_ticker = None
        self.current_tf = None

        self.timezone_dict = {'Amsterdam': tz.gettz('Europe/Amsterdam'),
                              'Chicago': tz.gettz('US/Central'),
                              'Frankfurt': tz.gettz('Europe/Berlin'),
                              'Hong Kong': tz.gettz('Asia/Hong_Kong'),
                              'London': tz.gettz('Europe/London'),
                              'Madrid': tz.gettz('Europe/Madrid'),
                              'Milan': tz.gettz('Europe/Rome'),
                              'MT4 Server Time': MT4ServerTime(),
                              'New York': tz.gettz('US/Eastern'),
                              'Paris': tz.gettz('Europe/Paris'),
                              'Shanghai': tz.gettz('Asia/Shanghai'),
                              'Singapore': tz.gettz('Asia/Singapore'),
                              'Sydney': tz.gettz('Australia/Sydney'),
                              'Tokyo': tz.gettz('Asia/Tokyo'),
                              'Zurich': tz.gettz('Europe/Zurich')
                              }

        self._define_trading_hours_dict()

        logging.info(f"API object created, with {len(self.ticker_lists)} predefined lists of tickers "\
                     f"and {len(self.methods)} API methods")
        logging.info(f"ticker lists:\n{[key for key in self.ticker_lists.keys()]}")
        logging.info(f"methods:\n{[fct.__name__ for fct in self.methods]}")

    def _create_api2source_timeframe_dict(self):
        """
        Creates the api2source_timeframe_dict attribute for AxitraderAPI

        Load axitrader timeframe dictionary from json file and creates the attribute  api2source_timeframe_dict

        Technical Note:
            uses pkg_resources to point to package directory

        JSON format:
            "D": {
                    "FX": {
                    "name": "1440",
                    "url-param": "1440"
                    },
                    "STOCK": {
                    "name": "1440",
                    "url-param": "1440"
                    }
            },

        """
        p2json_file = PACKAGE_ROOT / 'assets/json/axitrader-timeframes.json'
        if p2json_file.is_file():
            with open(p2json_file, 'r') as fp:
                self.api2source_timeframe_dict = json.load(fp)
        else:
            raise ValueError(f"No json file with name <{p2json_file.name}> at {p2json_file.absolute()}")

    def _define_ticker_lists(self):
        """
        Define standard lists of tickers for AxitraderAPI
        """
        self.ticker_lists:dict[str,list] = dict()
        self.ticker_lists['forex'] = sorted([subdict['tickers'][self.source]
                                             for _, subdict in self.ticker_dictionary.items()
                                             if self.source in subdict['tickers'].keys()
                                             and subdict['type'] == 'Margin FX Contract'])
        # self.ticker_lists['stock'] = sorted([subdict['tickers'][self.source]
        #                                      for _, subdict in self.ticker_dictionary.items()
        #                                      if self.source in subdict['tickers'].keys()
        #                                      and subdict['type'] == 'Stock in major index'])
        self.ticker_lists['index'] = sorted([subdict['tickers'][self.source]
                                             for _, subdict in self.ticker_dictionary.items()
                                             if self.source in subdict['tickers'].keys()
                                             and subdict['type'] == 'Index Cash CFDs'])
        self.ticker_lists['commodities'] = sorted([subdict['tickers'][self.source]
                                                   for _, subdict in self.ticker_dictionary.items()
                                                   if self.source in subdict['tickers'].keys()
                                                   and subdict['type'] == 'Commodity Cash CFDs'])
        self.ticker_lists['index futures'] = sorted([subdict['tickers'][self.source]
                                                     for _, subdict in self.ticker_dictionary.items()
                                                     if self.source in subdict['tickers'].keys()
                                                     and subdict['type'] == 'Index Future CFDs'])
        self.ticker_lists['commodities futures'] = sorted([subdict['tickers'][self.source]
                                                           for _, subdict in self.ticker_dictionary.items()
                                                           if self.source in subdict['tickers'].keys()
                                                           and subdict['type'] == 'Commodity Future CFDs'])
        self.ticker_lists['active'] = []
        for k, v in self.ticker_lists.items():
            if k != 'active':
                self.ticker_lists['active'].extend(v)
        self.ticker_lists['active'] = list(set(self.ticker_lists['active']))

    def _define_trading_hours_dict(self):
        """
        Define the dictionary of trading hours information for AxitraderAPI and the time zone dictionary
        """
        tickers_with_specific_trading_hours = self.ticker_lists['index'] + \
                                              self.ticker_lists['commodities'] + \
                                              self.ticker_lists['index futures'] + \
                                              self.ticker_lists['commodities futures']
        self.trading_hours_dict = dict()
        for ticker in tickers_with_specific_trading_hours:
            ticker_src = self.source2ticker_name[ticker]
            subdict = dict()
            for key in ["Trading Open", "Trading End", "Time Zone", "Weekmask"]:
                subdict[key] = self.ticker_dictionary[ticker_src][self.source].get(key, None)
            self.trading_hours_dict[ticker] = subdict

    def rename_mt4_price_files(self):
        """
        Rename all new price files, i.e. files with no date range in the name.

        Select all files in the self.raw_data folder which do not have a data range (fname ends w/ a MT4 time frame)
        Read the file, drop the last bar (which is often not fully formed), extract the start and end date, then
        rename the file into <ticker><mt4 timeframe> - from-startdate-to-enddate.csv
        """
        pattern = re.compile(r'^.*(1|5|15|30|60|240|1440)(?<!-from-\d{4}-\d{2}-\d{2}-to-\d{4}-\d{2}-\d{2}).csv$')
        new_price_files = [f for f in self.raw_data.glob('*.csv') if pattern.match(f.name) is not None]

        logging.info(f"Starting renaming {len(new_price_files)} files")

        for i, file in enumerate(new_price_files):
            with open(file, 'r') as f:
                content = f.readlines()
                # remove the last bar of the file as it is not a complete bar
                content = content[: -1]
                # extract dates from price file
                time_string = 'from-' + str(content[0][0:10]) + '-to-' + str(content[-1][0:10])
                time_string = time_string.replace('.', '-')

            new_file_name = self.raw_data / f"{file.stem}-{time_string}.csv"

            try:
                file.rename(new_file_name)
                with open(new_file_name, 'w') as f:
                    f.writelines(content)
                logging.info(f"{i+1}. Renamed {file.name} into {new_file_name.name} and removed last line")
            except:
                msg = f"{i+1}. {new_file_name} already exists. Could not rename {file.name}"
                logging.warning(msg)

        # Compare tickers in self.ticker_lists['active'] and tickers in the raw-data folder
        tickers_from_files = [self._extract_ticker(f.name) for f in self.raw_data.iterdir()
                              if f.is_file() and f.suffix == '.csv']
        tickers_all = sorted(list(set(tickers_from_files)))

        print(f"Total of {len(tickers_all)} tickers in historical price folder <{self.raw_data}> ")
        print(f"Total of {len(self.ticker_lists['active'])} active tickers for {self.source}")
        print('Tickers is in historical price folder but not in active list:')
        print(sorted(list(set(tickers_all).difference(set(self.ticker_lists['active'])))))
        print('Tickers is in active list but not in in historical price folder:')
        print(sorted(list(set(self.ticker_lists['active']).difference(set(tickers_all)))))

    @staticmethod
    def _extract_ticker(string):
        """Extract the ticker name price file names in the raw data folder"""
        m = re.match(r'(^.*)(60|240|1440|10080)-.*', string)
        if m is None:
            msg = f'No Match for {string}. Check the regex pattern for possible missing timeframes'
            raise NameError(msg)
        else:
            return m.group(1)

    def get_dataset_filename(
        self, 
        ticker:str, 
        tf:str, 
        pr_type=None
        ) -> str:
        """
        Returns the file name for a dataset according to the passed arguments

        Args:
            ticker (str):   Ticker for the dataset to retrieve
            tf (str):       TimeFrame
            pr_type (str):  Type of price. Not used for Axitrader. Kept for compatibility with general functions
        Returns:
            Filename formed from the passed arguments, as a str
        """

        return f"{self.source}-{ticker}-{self.get_time_frame_name(ticker, tf)}.csv"

    def get_raw_price_files(self, ticker_filter=None, timeframe_filter=None, time_filter=None)->list:
        """
        Retrieve the raw price files from the raw_data folder

        Returns:
            The list with all files Paths
        """
        ticker_filter = '*' if ticker_filter is None else ticker_filter
        timeframe_filter = '*' if timeframe_filter is None else timeframe_filter
        time_filter = '' if time_filter is None else time_filter

        raw_selection_pattern = f"{ticker_filter}{timeframe_filter}{time_filter}*.csv"
        selection_pattern = self.remove_multiple_wildcards(raw_selection_pattern)
        target_files_list = list(self.raw_data.glob(selection_pattern))

        return target_files_list

    def get_trading_hours(
        self, 
        ticker:str, # Ticker for the instrument (as define by the source Axitrader)
        calendar=None
        ) -> tuple[CustomBusinessHour, tzinfo]:
        """Retrieves Trading Hours dateoffset and Exchange Time Zone for the ticker, returns trading_hours, exchange_tz

        TODO: review API for calendar and possible modify (e.g. retrieve calendar from instrument info)
        """
        if calendar is None: calendar = USTradingHolidays
        ticker_src = self.source2ticker_name[ticker]
        subdict = dict()
        for key in ["Trading Open", "Trading End", "Time Zone", "Weekmask"]:
            subdict[key] = self.ticker_dictionary[ticker_src][self.source].get(key, None)

        if subdict['Trading Open'] is None:
            trading_open = tm(0, 1)
            trading_end = tm(23, 58)
            time_zone = MT4ServerTime()
            weekmask = 'Mon Tue Wed Thu Fri'
        else:
            trading_open = datetime.strptime(subdict['Trading Open'], '%H:%M').time()
            trading_end = datetime.strptime(subdict['Trading End'], '%H:%M').time()
            time_zone = self.timezone_dict.get(subdict["Time Zone"], MT4ServerTime())
            weekmask = subdict['Weekmask']

        trading_hours = CustomBusinessHour(start=trading_open,
                                           end=trading_end,
                                           weekmask=weekmask,   # type: ignore
                                           calendar=calendar()  # type: ignore
                                           )  
        
        return trading_hours, time_zone

    def read_csv_price_file(self, p2file):
        """
        Read historical price csv file into a Dataframe, according the the format corresponding to the API: Axitrader

        Args:
            p2file (Path):  path the the file to read into DataFrame

        Returns:
            DataFrame with the prices
        """
        df = pd.read_csv(p2file,
                         sep=',',
                         header=None,
                         index_col=[0],
                         parse_dates={'timestamp': [0, 1]},
                         na_values=['nan', 'null', 'NULL'],
                         keep_date_col=False
                         )
        df.columns = self.get_dataset_column_names()[1:]
        return df

    def update_price_datasets(
        self, 
        ticker_list:Optional[list[str]]=None,       #
        timeframe_list:Optional[list[str]]=None,    #
        price_type:Optional[str]=None,              # value in ['adjusted', 'non adjusted', 'both']; default value is 'both
        timefilter:Optional[str]=None                # used to filter the files to use for updating the dataset.
        ) -> None:              
        """
        Update all datasets for tickers in `ticker_list` and timeframe in `timeframe_list`.
        By default will update both adjusted and non adjusted prices datasets.
        
        """
        if timeframe_list is None:
            timeframe_list = ['D', 'H1']
        if ticker_list is None:
            ticker_list = self.ticker_lists['active']
        if price_type is None:
            price_type = 'adjusted'
        if timefilter is None:
            timefilter = ''

        for ticker in ticker_list:
            self.current_ticker = ticker
            for timeframe in timeframe_list:
                self.current_tf = timeframe
                self.adjusted = True 

                logging.info(f'Handling {ticker} for {timeframe}')
                dataset_file_name = self.get_dataset_filename(ticker, timeframe)
                path2dataset = self.datasets / dataset_file_name

                price_dict = self.get_price_dict(
                    ticker_filter=ticker,
                    timeframe_filter=self.get_time_frame_name(ticker, timeframe),
                    time_filter=timefilter)

                list_of_files = list(price_dict)

                if len(list_of_files) == 0:
                    logging.warning(f"No {timeframe} prices for {ticker}. Skipping to next")
                    break

                if path2dataset.exists():
                    logging.info(f' - Loading existing dataset from {dataset_file_name}')
                    full_ds = self.get_dataset(ticker=ticker, tf=timeframe)

                else:
                    logging.info(f' - Creating a new dataset for {dataset_file_name}')

                    # columns do not include the timestamp, which is the name of the index
                    full_ds = pd.DataFrame(columns=self.get_dataset_column_names()[1:])

                for file_name, df in price_dict.items():

                    # Crop all historical prices for bars prior to the earliest bar 
                    earliest_bar = self.ds_info_dictionary[self.source2ticker_name[ticker]]['earliest bar'][self.source]
                    filtered_df = df.loc[earliest_bar:].copy()

                    # Handle all duplicated bars
                    df = self.handle_duplicated_bars(df)

                    # Handle bars outside normal trading hours
                    df = self.normalize_trading_hours(df)

                    # Filter bars to avoid duplicated bars in dataset. 
                    filtered_df = self.exclude_idx_from_df(
                        df_with_idx_to_exclude=full_ds,
                        df_to_filter=df
                        )
                    logging.info(f' - Updating with {file_name}')
                    logging.info(
                       f'   Dataset: {full_ds.shape[0]}. File: {df.shape[0]}. Added {filtered_df.shape[0]} rows.')
                    full_ds = pd.concat([full_ds, filtered_df], verify_integrity=True, sort=False)
                    full_ds.sort_index(ascending=True, inplace=True)

                assert all(full_ds.isna()) is True, f'Some values in the dataset (full_ds) are NaN.'

                logging.info(f' - Saving updated dataset into {dataset_file_name}')
                full_ds.to_csv(path_or_buf=self.datasets / dataset_file_name, index_label='timestamp')

                # Update the time range info in ds_info_dictionary
                time_range = (full_ds.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                              full_ds.index[-1].strftime('%Y-%m-%d %H:%M:%S'))
                self.ds_info_dictionary[self.source2ticker_name[ticker]]['time ranges'][self.source][timeframe] = time_range

        # Update datasets_info.json based updated ds_info_dictionary (use pkg_resources to point to package directory)
        p2json_file = Path(pkg_resources.resource_filename(__name__, 'datasets_info.json'))
        if p2json_file.is_file():
            with open(p2json_file, 'w') as fp:
                json.dump(self.ds_info_dictionary, fp, indent=4)
        else:
            raise ValueError(f"No json file with name <{p2json_file.name}> at {p2json_file.absolute()}")

    def handle_duplicated_bars(self, df) -> pd.DataFrame:
        """Merge duplicated bars in dataframe df and returned df with no duplicated bar

        Method: 
            Open is the min(open) of 
            High = 
        """
        pass
        return df

    def normalize_trading_hours(self, df) -> pd.DataFrame:
        pass
        return df

    def date_latest_file_download(self):
        """Return the date of the last time pricing info was uploaded"""
        print('Not currently implemented for Axitrader')
        return datetime(year=1900, month=1, day=1)

    def analyze_price_consistency(
        self, 
        ticker:str, 
        tf='60', 
        date_filter=None, 
        problem_mask=None
        ) -> tuple[pd.DataFrame, CustomBusinessHour, tzinfo]:
        """
        TODO: docstring
        """

        if ticker is None:
            raise ValueError(f"ticker must be a string, not {ticker}")
        elif ticker not in self.ticker_lists['active']:
            raise ValueError(f"{ticker} is not a know value. Please select a ticker our of {self.ticker_lists['active']}")

        freq = '60T'
        trading_hours, exchange_tz = self.get_trading_hours(ticker)

        # load price dictionary for the ticker and pick the longest data range
        # TODO: update this to handle a merge set of prices
        price_dict = self.get_price_dict(ticker_filter=ticker, timeframe_filter=tf)
        list_nbr_rows = [d.shape[0] for d in price_dict.values()]
        idx_largest_df = list_nbr_rows.index(max(list_nbr_rows))
        fname, df = list(price_dict.items())[idx_largest_df]
        print(f"Handling {fname}")
        print(f"Trading Hours: {trading_hours.start[0]} <=> {trading_hours.end[0]}") # type: ignore
        print(f"From {df.index[0].strftime('%Y-%m-%d %H%M')} to {df.index[-1].strftime('%Y-%m-%d %H%M')}")

        # Create full range dataframe and cols to show
        full_df = self.create_full_range_price_set(df, freq=freq, trading_hrs=trading_hours, xch_tz=exchange_tz)
        earliest, latest = full_df.index[0], full_df.index[-1]
        print(f"Full range from {earliest.strftime('%Y-%m-%d %H%M')} to {latest.strftime('%Y-%m-%d %H%M')}")

        cols = ['dtIndex', 'MT4_Day', 'MT4', 'XCH', 'Day', 'Mth', 'Wk', 'D', 'Hr', 'Min', 'DST', 'Close', 'Trading']

        if date_filter is None:
            dt_filter = slice(earliest, earliest + timedelta(days=2))
        else:
            dt_filter = date_filter
        display(full_df.loc[dt_filter, cols])

        # Reviews bars with problems
        # No price data when Trading is active or price data exists when trading should be closed
        mask = (full_df['Trading'] & full_df['Close'].isna()) | (~full_df['Trading'] & ~full_df['Close'].isna())

        if date_filter is None:
            dt_filter = slice(earliest, latest)
        else:
            dt_filter = date_filter

        print(f"full_df shape: {full_df.shape}")
        self.analyze_problems(full_df, problem_mask=mask, date_filter=date_filter, cols_to_review=None)

        print(f"{100 * '-'}")
        print('All bars which have a small volume compared to the dataset average volume')

        col = 'Volume'
        cols_2 = cols + [col]
        avg_volume = full_df[col].mean()
        threshold_pct = 0.050
        print(f"Average Volume: {avg_volume:1.1f}. Threshold: {threshold_pct * avg_volume:1.1f}")
        display(full_df.loc[dt_filter, :].loc[
                    full_df.loc[dt_filter, col].apply(lambda x: x < threshold_pct * avg_volume), cols_2])

        print(f"{100 * '-'}")
        print('All bars with an original datetime (in mt4) with minutes not 0, or trading hours start or trading hours ',
              'end. These bars are assumed to be invalid. ',
              'Can compare with other aspects, such as Trading=False or low volume ...')

        col = 'dtIndex'
        values = [13,  # to catch the datetime nan in full_df
                  0,
                  trading_hours.start[0].minute, (trading_hours.start[0].minute - 1)%60, # type: ignore
                  trading_hours.end[0].minute, (trading_hours.end[0].minute + 1)%60]     # type: ignore
        print(f"List of valid values for minutes: {set(values[1:])}")

        display(full_df.loc[dt_filter, :].loc[
                    full_df.loc[dt_filter, col].apply(lambda x: x.minute not in values), cols + ['Volume']])

        return full_df, trading_hours, exchange_tz

    @staticmethod
    def create_full_range_price_set(
        df:pd.DataFrame,        # historical price df w/ DatetimeIndex and cols: Open, Low, High, Close, Volume
        freq:Optional[str]=None,# frequency used for PeriodIndex conversion
        trading_hrs=None,       # DateOffset representing trading hours, typically CustomBusinessHours
        xch_tz=None,            # time zone for the exchange where the instrument is traded
        calendar=None
        ) -> pd.DataFrame:      # full range df
        """ Create a full range df based on trading hours and calendar for consistency preparation"""

        # Default arguments and argument checks
        # validate freq
        freq_d = {'H': '60T','60min': '60T','60T': '60T',
                  # 'D': '24H','B': '24H','24H': '24H'
                 }
        valid_freq = freq_d.keys()
        if freq is None: freq = '60T'
        if freq in valid_freq: freq = freq_d[freq]
        else:
            raise ValueError(f"{freq} is not valid for freq. Please use one of {list(valid_freq)}")

        if trading_hrs is None:
            trading_open = tm(9, 0)
            trading_end = tm(16, 59)
            weekmask = 'Mon Tue Wed Thu Fri'
            trading_hrs = CustomBusinessHour(
                    start=trading_open, 
                    end=trading_end,
                    weekmask=weekmask,  # type: ignore
                    calendar=calendar() # type: ignore
                    )

        if xch_tz is None:
            raise ValueError(f"exchange_tz must be a valid tz timezone")

        if calendar is None: 
            calendar = USTradingHolidays

        def match_dtidx_to_period(dtidx, trading_hrs):
            """Matches a DatetimeIndex value to a PeriodIndex value compatible with trading_hrs offset

            Used to create a PeriodIndex value (of full index) from a DatetimeIndex value (of full index) in order
            to allows the transfer of data from a df with DatetimeIndex to a df with PeriodIndex. It uses the start
            time of the dateoffest trading_hrs. The start of the period is compatible with the

            Args:
                dtidx (pd.DatetimeIndex):   the DatatimeIndex value to convert into a PeriodIndex value
                trading_hrs (DateOffset):   the applicable DateOffset from which start will be extracted

            Returns: pidx (pd.PeriodIndex): the PeriodIndex value corresponding to the passed dtidx
            """
            trading_open_minutes = trading_hrs.start[0].minute
            if dtidx.minute < trading_open_minutes:
                dtidx = dtidx - timedelta(hours=1)
            return dtidx.replace(minute=trading_open_minutes).to_period(freq=freq)

        mt4 = MT4ServerTime()

        # Extract start and end time from trading_hrs. Note: trading_hrs.start is a tuple (datetime, ..)
        s_time = trading_hrs.start[0]   # type: ignore
        e_time = trading_hrs.end[0]     # type: ignore

        earliest = min(df.index[0], df.index[0].replace(hour=s_time.hour, minute=s_time.minute))
        latest = max(df.index[-1], df.index[-1].replace(hour=e_time.hour, minute=e_time.minute))
        print(earliest, '<<<===>>>', latest)

        # Make a temporary copy of df and add a copy of its DatetimeIndex
        tdf = df.copy()
        tdf['dtIndex'] = tdf.index.copy()

        # Create the full date range DataFrame (from earliest to latest)
        idx_full = pd.period_range(start=earliest, end=latest, freq=freq)
        fdf = pd.DataFrame(index=idx_full, columns=tdf.columns)
        fdf.index.name = 'Period (MT4)'

        # Map df DatetimeIndex into a PeriodIndex compatible with idx_full, and insert it into tdf.index
        how = 'end'
        period_idx = tdf.index.map(partial(match_dtidx_to_period, trading_hrs=trading_hrs))
        tdf.index = period_idx

        # Copy all rows of tdf into fdf where the PeriodIndex match
        fdf.loc[period_idx, :] = tdf

        # Create additional features to fdf
        fdf = fdf.assign(MT4_dt=lambda d: d.index.to_timestamp(freq=freq,
                                                               how=how).tz_localize(tz=mt4),
                         MT4_Day=lambda d: d.index.to_timestamp(freq=freq,
                                                                how=how).tz_localize(tz=mt4).day_name(),
                         XCH_dt=lambda d: d.index.to_timestamp(freq=freq,
                                                               how=how).tz_localize(tz=mt4).tz_convert(tz=xch_tz),
                         Day=lambda d: d.index.to_timestamp(freq=freq,
                                                            how=how).tz_localize(tz=mt4).tz_convert(tz=xch_tz).day_name(),
                         Mth=lambda d: d.index.to_timestamp(freq=freq,
                                                            how=how).tz_localize(tz=mt4).tz_convert(tz=xch_tz).month,
                         D=lambda d: d.index.to_timestamp(freq=freq,
                                                          how=how).tz_localize(tz=mt4).tz_convert(tz=xch_tz).day,
                         Hr=lambda d: d.index.to_timestamp(freq=freq,
                                                           how=how).tz_localize(tz=mt4).tz_convert(tz=xch_tz).hour,
                         Min=lambda d: d.index.to_timestamp(freq=freq,
                                                            how=how).tz_localize(tz=mt4).tz_convert(tz=xch_tz).minute
                         )
        fdf = fdf.assign(Wk=lambda x: x['XCH_dt'].dt.isocalendar().week,
                         MT4=lambda x: x['MT4_dt'].dt.strftime("%b-%d %H:%M"),
                         XCH=lambda x: x['XCH_dt'].dt.strftime("%b-%d %H:%M"),
                         Trading=lambda x: x['XCH_dt'].apply(lambda d: trading_hrs.is_on_offset(d)),
                         DST=lambda x: x['MT4_dt'].apply(lambda d: d.dst())
                         )

        # Cast dtypes of columns with NaN to correct type
        fdf['dtIndex'] = fdf['dtIndex'].fillna(value=datetime(1900, 1, 2, 13, 13))
        fdf_dtypes = {'Open': 'float',
                      'High': 'float',
                      'Low': 'float',
                      'Close': 'float',
                      'Volume': 'float',
                      'dtIndex': 'datetime64[ns]',
                      'Trading': 'bool',
                      }

        fdf.astype(fdf_dtypes)

        return fdf

    @staticmethod
    def show_weeks_with_problems(df, problem_mask=None, freq='H', map_to_show=None, show_summary=True):
        """ Obsolete. Replaced by self.analyse_problems()"""
        raise NotImplementedError(f"This method is obsolete and is replaced by .analyze_problems()")

    @staticmethod
    def analyze_problems(df, problem_mask=None, date_filter=None, cols_to_review=None, freq='60T'):
        """Provides a report on the rows in df that have problems, based on the passed problem_mask

        todo: finish this docstring
        Args:
            df (pd.DataFrame:
            problem_mask (list(bool)):
            date_filter (str):
            cols_to_review (list(str)):
            freq (str):

        Returns:    None
        """
        if 'Close' not in df.columns:
            raise ValueError('df must have a columns <Close>')

        if problem_mask is None:
            if 'Trading' in df.columns and 'Close' in df.columns:
                problem_mask = ((df['Trading'] & df['Close'].isna()) | (~df['Trading'] & ~df['Close'].isna()))
            else:
                raise ValueError('Provide a problem_mask')

        if date_filter is None:
            date_filter=slice(df.index[0], df.index[-1])

        if cols_to_review is None:
            cols_to_review = ['Mth', 'Wk', 'Day', 'D', 'Hr', 'Min']

        year_first = df.loc[df.index[0], 'XCH_dt'].year
        year_last = df.loc[df.index[-1], 'XCH_dt'].year

        # create df with  all rows with problems
        row_with_problems = df.loc[problem_mask, :]

        bins = {'Mth': list(range(1, 14)),
                'Wk': list(range(1, 54)),
                'Day': list(range(0, 5)),
                'D': list(range(1, 33)),
                'Hr': list(range(1, 26)),
                'Min': list(range(0, 61, 10))
                }

        column_d = {'Mth': list(range(1, 13)),
                    'Wk': list(range(1, 53)),
                    'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    'D': list(range(1, 32)),
                    'Hr': list(range(1, 25)),
                    'Min': None
                    }

        for col in cols_to_review:
            with_problems = sorted(list(row_with_problems.loc[date_filter, col].unique()))
            problems_counts = row_with_problems.loc[date_filter, col].value_counts()

            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            row_with_problems.loc[date_filter, col].hist(ax=ax, rwidth=0.8, align='left', bins=bins[col], log=False)
            if isinstance(date_filter, slice):
                title = f"{col} with problems from {year_first} to {year_last}"
            else:
                title = f"{col} with problems in {date_filter}"
            ax.set_title(title)
            plt.show()

            if column_d[col] is not None:
                idx = [str(yr) for yr in range(year_first, year_last + 1)]
                problems_tbl = pd.DataFrame(index=idx, columns=column_d[col])
                for yr in problems_tbl.index:
                    # yr_problems_counts = row_with_problems.loc[str(yr), col].value_counts()
                    problems_tbl.loc[yr, :] = row_with_problems.loc[str(yr), col].value_counts()

                total = pd.DataFrame(index=['Total'], columns=column_d[col])
                total.loc['Total', :] = row_with_problems.loc[:, col].value_counts()
                problems_tbl = pd.concat([problems_tbl,total])

                display(problems_tbl.fillna(value=0))
            else:
                print(with_problems)
                print([problems_counts.get(idx) for idx in with_problems])

    @staticmethod
    def get_dataset_column_names():
        """
        Return the name of dataset columns depending on the ticker

        Implementation for Axitrader API. All datasets files have the same format

        Returns: list of dataset columns names
        """
        return ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']

    # @staticmethod
    # def get_price_file_correct_header_line():
    #     """
    #     Retrieves correct price file header based on the ticker.
    #
    #     Implementation for Axitrader API. All price files have the same format
    #
    #     Returns: list of header names
    #     """
    #     return ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']


class QuandlAPI(API):
    """API for Quandl"""

    source = 'quandl'
    folder = 'quandl'
    data_format = 'quandl'

    def __init__(self):
        """Specific initialisation for Alphavantage"""
        super().__init__()
        self.api_key_ring = ApiKeyRing(config=self.config, source=self.source)
        self.throttle_delay = 2 + 20 / len(self.api_key_ring)  # Max API frequency is 5 calls/min and 500 calls/day

        self.methods = [self.info, self.info_methods,
                        self.retrieve_prices,
                        self.get_dataset, self.update_price_datasets,
                        self.date_latest_file_download,
                        self.switch_to_HDD, self.switch_to_NAS]

        self.current_ticker = None
        self.current_tf = None

        # logging.info(f"API object created, with {len(self.ticker_lists)} predefined lists of tickers "\
        #              f"and {len(self.methods)} API methods")
        # logging.info(f"ticker lists:\n{[key for key in self.ticker_lists.keys()]}")
        # logging.info(f"methods:\n{[fct.__name__ for fct in self.methods]}")

    def _define_ticker_lists(self):
        """
        Define standard lists of tickers for AlphavantageAPI
        """
        self.ticker_lists = dict()
        self.ticker_lists['active'] = []

    def _create_api2source_timeframe_dict(self):
        """
        Creates the api2source_timeframe_dict attribute for QuandlAPI
        FIXME: this implementation is not correct, correct it.

        Load Quandl timeframe dictionary from json file and creates the attribute  api2source_timeframe_dict

        Technical Note:
            uses pkg_resources to point to package directory

        JSON format:
            "D": {
                    "FX": {
                    "name": "1440",
                    "url-param": "1440"
                    },
                    "STOCK": {
                    "name": "1440",
                    "url-param": "1440"
                    }
            },

        """

        p2json_file = Path(pkg_resources.resource_filename(__name__, 'axitrader-timeframes.json'))
        if p2json_file.is_file():
            with open(p2json_file, 'r') as fp:
                self.api2source_timeframe_dict = json.load(fp)
        else:
            raise ValueError(f"No json file with name <{p2json_file.name}> at {p2json_file.absolute()}")

    def build_url(self, ticker_name, tf, start_date=None, end_date=None):
        """Build the URL for alphavantage API, based on passed parameters

        https://www.quandl.com/api/v3/datasets/WIKI/AAPL.csv?start_date=1985-05-01&end_date=1997-07-01&order=asc&api_key=YOUR_API_KEY_HERE

        """
        # FIXME: method not implemented yet
        raise NotImplementedError('Method not fully implemented yet')
        symbol_str = str(ticker_name)

        api_key = self.api_key_ring.get_new_key()

        url = 'https://www.quandl.com/api/v3/datasets/WIKI/' + symbol_str + 'csv?order=asc' \
              + '&apikey=' + api_key + 'start_date=' + start_date + '&end_date=' + end_date

        return url

    def get_price_file_correct_header_line(self, ticker):
        """Retrieves the correct file header based on the ticker. When passed symbol is not a ticker, returns None"""
        # FIXME: method not implemented yet

    def handle_file_error_message(self, filepath):
        """Check that downloaded file is valid. Otherwise, handle API quota errors

        E.g. proper headers must be: 'timestamp,open,high,low,close,volume'
            "Error Message": "Invalid API call. Please retry or visit the documentation

        :param      filepath: name of the file to test
        :return:    True is no error, False is error
        """
        # FIXME: method not implemented yet
        raise NotImplementedError('Method not fully implemented yet')

    def handle_api_error(self):
        """Handles keys when an API quota error is detected

        If there are enough keys, removes the key under which the quota error was raised.
        Throttle delay is also adjusted to match the number of keys.
        If there are not enough keys, reinitialize the full set of keys and pauses for 5 min.
        """
        # FIXME: method not implemented yet
        raise NotImplementedError('Method not fully implemented yet')

    def throttle_api(self):
        """Execute the throttle actions. Current implementation is a simple sleep time."""
        # FIXME: method not implemented yet

    def retrieve_prices(self, mode='compact', tickers=None, timeframes=None,
                        skip_existing=False, verbose=False):
        """Get instrument pricing data from alphavantage for default tickers and tf or specified ones.

        mode:                       str 'full' or 'compact'
                                    default = 'compact
        tickers: (optional)         list or str: ['EURUSD', 'USDJPY', 'MSFT', 'RSA.LON'].
                                    default is full list of tickers
        timeframes: (optional)      list: Subset of ['M', 'W', 'D', 'H1', 'M30', 'M15', 'M5', 'M1']
                                    default is ['D', 'H1', 'M30']
        skip_existing: (optional)   boolean. If True, skip the download of a file in case of pre-existing file with the
                                    same name. Used to speed up re-updated of historical data when done the same day.
                                    default is False
        verbose: (optional)         boolean: True to log info in console.

        return:                  None
        """
        # FIXME: method not implemented yet
        raise NotImplementedError('Method not fully implemented yet')

    def get_dataset_column_names(self, ticker):
        """Return the name of dataset columns depending on the ticker

        Currently implemented: alphavantage FX and STOCK
        TODO: implement instrument types for other sources: MT4 FX, STOCK, commodity, index, ...
        """
        # Implementation for Alphavantage
        instrument_type = self.get_instrument_type(ticker)
        if instrument_type == 'FX':
            return ['timestamp', 'Open', 'High', 'Low', 'Close']
        elif instrument_type == 'STOCK':
            return ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        else:
            raise ValueError(f"{instrument_type} is not a known instrument type for {self.source}")
            return False
        # FIXME: method not implemented yet
        raise NotImplemented('Method not fully implemented yet')

    def update_price_datasets(self, ticker_list=None, timeframe_list=None, timefilter=None):
        """Not tested yet

        ticker_list:
        timeframe_list:
        timefilter:

        TODO: test this method (update price datasets)
        """
        if timeframe_list is None:
            timeframe_list = ['D', 'H1', 'M30']
        if ticker_list is None:
            ticker_list = self.ticker_lists['forex']
        if timefilter is None:
            timefilter = ''

        for ticker in ticker_list:
            for timeframe in timeframe_list:
                logging.info(f'Handling {ticker} for {timeframe}')
                instrument_type = self.get_instrument_type(ticker)
                dataset_file_name = f'{self.source}-{ticker}-{self.get_time_frame_name(ticker, timeframe)}.csv'
                path2dataset = self.datasets / dataset_file_name

                price_dict = self.get_price_dict(
                    ticker_filter=ticker,
                    timeframe_filter=self.get_time_frame_name(ticker, timeframe),
                    time_filter=timefilter,)
                list_of_files = list(price_dict)

                if len(list_of_files) == 0:
                    logging.warning(f"No {timeframe} prices for {ticker}. Skipping to next")
                    break

                if path2dataset.exists():
                    logging.info(f' - Loading existing dataset from {dataset_file_name}')
                    full_ds = pd.read_csv(path2dataset,
                                          sep=',',
                                          index_col=0,
                                          header=None,
                                          skiprows=1,
                                          parse_dates=[0],
                                          names=self.get_dataset_column_names(ticker),
                                          na_values=['nan', 'null', 'NULL']
                                          )
                else:
                    logging.info(f' - Creating a new dataset for {dataset_file_name}')
                    full_ds = pd.DataFrame(columns=price_dict[list_of_files[0]].columns)

                for file_name, df in price_dict.items():
                    filtered_df = self.exclude_idx_from_df(df_with_idx_to_exclude=full_ds,
                                                           df_to_filter=df.iloc[:-1, :])
                    logging.info(f' - Updating with {file_name}')
                    logging.info(
                       f'   Dataset: {full_ds.shape[0]}. File: {df.shape[0]}. Added {filtered_df.shape[0]} rows.')
                    full_ds = pd.concat([full_ds,filtered_df], verify_integrity=True, sort=False)
                    full_ds.sort_index(ascending=True, inplace=True)

                assert all(full_ds.isna()) is True, f'Some values in the dataset (full_ds) are NaN.'

                logging.info(f' - Saving updated dataset into {dataset_file_name}')
                full_ds.to_csv(path_or_buf=self.datasets / dataset_file_name, index_label='timestamp')
        # FIXME: method not implemented yet
        raise NotImplementedError('Method not fully implemented yet')


class YahooAPI(API):
    """
    API for Yahoo Finance

    This API will make use of yfinance package for scrapping information from the site and extent it with a set
    of utility methods
    """

    source = 'yahoo'
    folder = 'yahoo'
    data_format = 'yahoo'

    def __init__(self):
        """Specific initialisation for Yahoo Finance

        Attributes:
            ??? self.api_key_ring (ApiKeyRing):
            self.api2source_timeframe_dict:
            self.api_info_dict:
            self.source2ticker_name:
            self.current_ticker:
            self.current_tf:
            self.throttle_delay (int):
            self.ticker_lists:
            self.methods:
            self.adjusted:
            self.mode:

        """
        super().__init__()
        self.throttle_delay = 0

        self.methods = [self.info, self.info_methods,
                        # self.retrieve_prices, self.search_for_stock_info,
                        self.get_dataset, 
                        # self.update_price_datasets,
                        self.get_common_dateindex,
                        self.date_latest_file_download,
                        self.get_prices_from_several_datasets,
                        self.switch_to_HDD, self.switch_to_NAS]

        self.current_ticker = None
        self.current_tf = None
        self.adjusted = True
        self.mode = 'compact'


        # # make a translation dictionary between data source ticker names and ticker_dictionary names
        # self.source2ticker_name = dict([(subdict['tickers'][self.source], symbol)
        #                                 for symbol, subdict in self.ticker_dictionary.items()
        #                                 if self.source in subdict['tickers'].keys()])




        logging.info(f"API object created, with {len(self.ticker_lists)} predefined lists of tickers "\
                     f"and {len(self.methods)} API methods")
        logging.info(f"ticker lists:\n{[key for key in self.ticker_lists.keys()]}")
        logging.info(f"methods:\n{[fct.__name__ for fct in self.methods]}")

    def _create_api2source_timeframe_dict(self):
        """Load yahoo time frame dictionary from json file (use pkg_resources to point to package directory)"""
        p2json_file = Path(pkg_resources.resource_filename(__name__, 'yahoo-timeframes.json'))
        if p2json_file.is_file():
            with open(p2json_file, 'r') as fp:
                self.api2source_timeframe_dict = json.load(fp)
        else:
            raise ValueError(f"No json file with name <{p2json_file.name}> at {p2json_file.absolute()}")

    def _define_ticker_lists(self):
        # Define standard lists of tickers for this data source:
        self.ticker_lists = dict()
        self.ticker_lists['stock'] = [subdict['tickers'][self.source]
                                              for _, subdict in self.ticker_dictionary.items()
                                              if self.source in subdict['tickers'].keys()
                                              and subdict['type'] == 'Stock in major index']
        self.ticker_lists['index'] = [subdict['tickers'][self.source]
                                              for _, subdict in self.ticker_dictionary.items()
                                              if self.source in subdict['tickers'].keys()
                                              and subdict['type'] == 'Index Cash CFDs']
        self.ticker_lists['commodities'] = [subdict['tickers'][self.source]
                                              for _, subdict in self.ticker_dictionary.items()
                                              if self.source in subdict['tickers'].keys()
                                              and subdict['type'] == 'Commodity Cash CFDs']
        self.ticker_lists['index futures'] = [subdict['tickers'][self.source]
                                              for _, subdict in self.ticker_dictionary.items()
                                              if self.source in subdict['tickers'].keys()
                                              and subdict['type'] == 'Index Future CFDs']
        self.ticker_lists['commodities futures'] = [subdict['tickers'][self.source]
                                              for _, subdict in self.ticker_dictionary.items()
                                              if self.source in subdict['tickers'].keys()
                                              and subdict['type'] == 'Commodity Future CFDs']
        self.ticker_lists['active'] = []
        for k, v in self.ticker_lists.items():
            if k != 'active':
                self.ticker_lists['active'].extend(v)
        self.ticker_lists['active'] = list(set(self.ticker_lists['active']))

    def display_download_url(self, lists=None, markdown=True):
        """Print url to download historical data for instruments in passed lists
        
        By default, prints in Markdown format to save in a Markdown cell
        If markdown is False. prints the list of urls
        """
        if lists is None:
            lists = list(self.ticker_lists.keys())
            lists.remove('active')

        for l in lists:
            print(f"Instruments to download for '{l}'")
            if markdown:
                print(f"| # | Yahoo Symbol |  Download Page |")
                print(f"|:-:|:------------:|:----:|")
            for i, instrument in enumerate(self.ticker_lists[l]):
                s = urlparse.quote(instrument)
                if markdown:
                    print(f"|{i+1} | {instrument} |  [url](https://finance.yahoo.com/quote/{s}/history?p={s}) |")
                else:
                    print(f" {i+1:3d}. {instrument:.<15s} https://finance.yahoo.com/quote/{s}/history?p={s}")

    def date_latest_file_download(self):
        return datetime.now()

    def get_raw_price_files(self, ticker_filter=None, time_filter=None, timeframe_filter=None)->list:
        """Gets the list of filenames for all raw historical files matching filters, and return as list

        File name formats: `yahoo_symbol-YYYY-MM-DD-to-YYYY-MM-DD.csv` where first and last bar dates as YYYY-MM-DD

        ticker_filter (str):    yahoo symbol to use (e.g. ^GSPC)
        time_filter (datetime or str): date of the earliest last_bar_date to pick
        timeframe_filter:       not implemented and ignored
        """
   
        # regex pattern to extract symbols, first_bar and last_bar from previously saved files
        pat = re.compile(r'^(.*)-(\d{4}-\d{2}-\d{2})-to-(\d{4}-\d{2}-\d{2}).csv')
        target_files_list = [f for f in self.raw_data.iterdir() if re.search(pat, f.name)]
   
        if ticker_filter is not None:
            target_files_list = [
                f for f in target_files_list
                if re.search(pat, f.name).group(1) == ticker_filter  
            ]

        if time_filter is not None:
            if isinstance(time_filter, str):
                time_filter = self.string2date(time_filter)

            target_files_list = [
                f for f in target_files_list
                if self.string2date(re.search(pat, f.name).group(3)) >= time_filter  
            ]

        return target_files_list

    def handle_new_raw_price_files(self)->None:
        """Retrieves newly downloaded raw historical files and modify their name

        File name formats:
            file newly downloaded: `yahoo_symbol.csv`
            files after renaming:  `yahoo_symbol-YYYY-MM-DD-to-YYYY-MM-DD.csv` with first and last bar dates
        """
        # regex pattern to extract symbols from newly downloaded files
        pat_new = re.compile(r'^([\w\d\^\.]*).csv')
        fnames_new = [f for f in self.raw_data.iterdir() if re.search(pat_new, f.name)]

        for file in fnames_new:
            with open(file, 'r') as fp:
                lines = fp.readlines()
            
            first_bar = lines[1][:10].replace('.', '-')
            last_bar = lines[-1][:10].replace('.', '-')
            new_fname = f"{file.stem}-{first_bar}-to-{last_bar}.csv"

            print(f"renaming {file.name} into {(self.raw_data / new_fname).name}")
            os.rename(file, self.raw_data / new_fname)

    @staticmethod
    def string2date(str):
        return datetime.strptime(str,'%Y-%m-%d')

    @staticmethod
    def get_price_file_correct_header_line(ticker):
        """
        Retrieves correct price file header based on the ticker.

        Implementation for Yahoo API. All price files have the same format"""
        return ['timestamp', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    @staticmethod
    def get_dataset_column_names(ticker):
        """
        Return the name of dataset columns depending on the ticker

        Implementation for Yahoo API. All datasets files have the same format"""
        return ['timestamp', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']




if __name__ == '__main__':
    # from pandas.tseries.offsets import BusinessDay, BusinessHour, Hour, CustomBusinessDay, CustomBusinessHour

    from datetime import datetime

    import pandas as pd
    from dateutil import tz

    NY = tz.gettz(name='America/New_York')
    FR = tz.gettz(name='Europe/Paris')
    MT4 = MT4ServerTime(verbose=True)

    # dt_1 = datetime(2022,1,1,16,59, tzinfo=NY)
    # print(dt_1)
    # _ = dt_1.astimezone(MT4) 
    # print(_)

    # dt_1 = datetime(2022,1,1,16,59, tzinfo=NY)
    # _ = dt_1.astimezone(MT4) 
    # print(_)
    
    dt_1 = pd.Timestamp(year=2022,month=5,day=1,hour=16,minute=59, tzinfo=NY)
    _ = dt_1.tz_convert(MT4)
    # _ = dt_1.tz_convert(FR)
    print(_)
    
    # dt_1 = pd.Timestamp(year=2022,month=5,day=1,hour=16,minute=59, tzinfo=NY)
    # _ = dt_1.astimezone(MT4)
    # print(_)
    # print('---')
    
    pass
