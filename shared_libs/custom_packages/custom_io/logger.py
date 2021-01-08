
import os
import sys
import time
import logging
import torch
from shared_libs.custom_packages.custom_basic.operations import chk_d


########################################################################################################################
# Customized logger
########################################################################################################################

class Logger(object):
    """
    A type of logger that is able to
        1. Writing:
            (1) Write different messages simultaneously to file & screen.
            (2) Write logs by appending existing logs.
        2. Generating & parsing formatted log line:
            Title: Epoch[%-3d], batch[%-3d] ---> Loss_name1: loss_value1; Loss_name2: loss_value2; ...
    """
    def __init__(self, log_dir, log_name, append_mode=False, **kwargs):
        # Configuration & data
        self._log_dir = log_dir
        self._log_name = log_name
        self._append_mode = append_mode
        if 'formatted_prefix' in kwargs.keys():
            self._formatted_prefix = kwargs['formatted_prefix']
        if 'formatted_counters' in kwargs.keys():
            self._formatted_counters = kwargs['formatted_counters']
            if isinstance(self._formatted_counters, str): self._formatted_counters = [self._formatted_counters]
            # Set a property for fetching
            setattr(self, 'formatted_counters', self._formatted_counters)
        # 1. Loggers redirect to disk file and screen respectively.
        self._log_identifier = None
        self._logger_file = None
        self._logger_screen = None
        # 2. Get loggers
        self._get_logger_file()
        self._get_logger_screen()

    def _get_logger_file(self):
        # Get log path
        if not os.path.exists(self._log_dir): os.makedirs(self._log_dir)
        # (1) Open stale logger
        if self._append_mode:
            # 1. Get stale
            # (1) Init
            stale_loggers_path = []
            # (2) Search
            for obj in os.listdir(self._log_dir):
                if obj.endswith("_%s.log" % self._log_name):
                    stale_loggers_path.append(os.path.join(self._log_dir, obj))
            # 2. Check & get result
            assert len(stale_loggers_path) == 1, "Too many matched stale loggers. "
            log_path = stale_loggers_path[0]
        # (2) Create new one
        else:
            log_path = os.path.join(
                self._log_dir, time.strftime('%Y%m%d%H%M%S_', time.localtime(time.time())) + self._log_name + '.log')
        # Set log_identifier based on log_path
        self._log_identifier = log_path
        # 1. Create logger
        self._logger_file = logging.getLogger(self._log_identifier)
        self._logger_file.setLevel(logging.INFO)
        # 2. Create file handler
        # (1) Create
        fh = logging.FileHandler(log_path, mode='a')
        fh.setLevel(logging.INFO)
        # (2) Define output format for handler
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        # (3) Add logger to handler
        self._logger_file.addHandler(fh)

    def _get_logger_screen(self):
        # 1. Create logger
        self._logger_screen = logging.getLogger('%s-to_screen' % self._log_identifier)
        self._logger_screen.setLevel(logging.INFO)
        # 2. Create screen handler
        # (1) Create
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # (2) Define output format for handler
        ch.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        # (3) Add logger to handler
        self._logger_screen.addHandler(ch)

    def info_individually(self, msg, to_screen=True):
        """
            Mode 1: Only writing to file (msg!=None & to_screen=False)
            Mode 2: Writing to file & printing the same msg to screen: (msg!=None & to_screen=True)
            Mode 3: Writing to file & printing different msg to screen: (msg!=None & to_screen=str)
            Mode 4: Only printing msg on screen: (msg=None & to_screen=str)
        :param msg:
        :param to_screen:
        :return:
        """
        # 1. Write to file.
        if msg is not None:
            self._logger_file.info(msg)
        # 2. Print to screen.
        # (1) The same message as written to file.
        if to_screen is True:
            assert msg is not None, "Message written to log file must be provided before printing them on screen. "
            self._logger_screen.info(msg)
        # (2) Not print.
        elif to_screen is False:
            pass
        # (3) Different message.
        else:
            assert isinstance(to_screen, str), "Invalid instance of 'to_screen'. Must be string. "
            self._logger_screen.info(to_screen)

    ####################################################################################################################
    # Formatted information
    ####################################################################################################################

    def info_formatted(self, counters=None, items=None, title=..., **kwargs):
        # Set title (formatted_prefix)
        if title is ...:
            title = self._formatted_prefix if hasattr(self, '_formatted_prefix') else None
        # Set msg
        # (1) File
        msg_file = self.generate_logger_info(None, counters, items)
        # (2) Screen
        #   Get items
        if 'no_display_keys' in kwargs.keys():
            if kwargs['no_display_keys'] == 'all': items_screen = {}
            else: items_screen = {
                key: items[key] for key in filter(lambda k: k not in kwargs['no_display_keys'], items.keys())}
        else:
            items_screen = items
        #   Set msg
        msg_screen = self.generate_logger_info(title, counters, items_screen) \
            if items_screen else False
        # Information
        self.info_individually(msg_file, to_screen=msg_screen)

    def generate_logger_info(self, title, counters, items):
        """
        :param title:
        :param counters: An ordered dict.
        :param items: a dict.
        :return: Loss information str.
            (Title: )(counter_1[%-3d], ..., counter_n[%-3d] ---> )Loss_name1: loss_value1; Loss_name2: loss_value2; ...
        """
        # 1. Init results.
        info = ''
        # 2. Get results
        ################################################################################################################
        # Before --->
        ################################################################################################################
        # (1) Title
        if title is not None:
            assert " - " not in title, " ' - ' is not allowed in the title. "
            info += '%s: ' % title
        # (2) Generate counters
        info_counters = ''
        #   Get str
        if counters is None:
            pass
        else:
            # 1. Get iterable
            if isinstance(counters, dict):
                iterable = []
                for k in self._formatted_counters:
                    try:
                        v = counters[k]
                    except KeyError:
                        v = None
                    iterable.append((k, v))
            else:
                # Formulate counters
                if isinstance(counters, int): counters = [counters]
                assert len(counters) <= len(self._formatted_counters)
                for _ in range(len(self._formatted_counters) - len(counters)): counters.append(None)
                # Iterable
                iterable = zip(self._formatted_counters, counters)
            # 2. Get info
            for index, (k, v) in enumerate(iterable):
                if v is None: continue
                info_counters += "%s[%-3d]" % (k, v)
                if index < len(counters) - 1:
                    info_counters += ", "
        #   Set str
        if info_counters != '': info_counters += ' ---> '
        #   Save
        info += info_counters
        ################################################################################################################
        # After --->
        ################################################################################################################
        # Generate for items
        for name, value in items.items():
            assert ";" not in name, "';' is not allowed in an item name. "
            # 1. Generate information
            if isinstance(value, int):
                msg = "%s: %d; " % (name, value)
            elif isinstance(value, float):
                msg = "%s: %7.4f; " % (name, value)
            elif isinstance(value, torch.Tensor):
                msg = "%s: %7.4f; " % (name, value.item())
            elif isinstance(value, str):
                assert ";" not in value, "';' is not allowed in an item value. "
                msg = "%s: %s; " % (name, value)
            elif value is None:
                continue
            else:
                raise ValueError
            # 2. Add to information
            info += msg
        # Return
        return info

    ####################################################################################################################
    # Utils
    ####################################################################################################################

    @staticmethod
    def parse_logger_line(line):
        # 1. Init result
        result = {}
        # 2. Parsing
        # Split by ' ---> ', for examples of different formats:
        #   year-month-day hour:minute:second - (counter[%-3d])( ---> )(name1: value1; )
        datetime, message = line.split(" - ")
        # --------------------------------------------------------------------------------------------------------------
        # Get datetime
        # --------------------------------------------------------------------------------------------------------------
        date, t = datetime.split()
        # Get raw str
        year, month, day = str(date).split("-")
        hour, minute, second = str(t).split(":")
        sec_main, sec_remain = str(second).split(",")
        # Get value
        year, month, day = int(year), int(month), int(day)
        hour, minute, second = int(hour), int(minute), float(sec_main) + float(sec_remain) / 1000.0
        # Save
        result.update({'year': year, 'month': month, 'day': day, 'hour': hour, 'minute': minute, 'second': second})
        # --------------------------------------------------------------------------------------------------------------
        # Get title & counters
        # --------------------------------------------------------------------------------------------------------------
        assert isinstance(message, str)
        if " ---> " not in message:
            items = message
        else:
            counters, items = message.split(" ---> ")
            for item in counters.split(", "):
                k, v = item.split("[")
                result[k] = int(v.split("]")[0])
        # --------------------------------------------------------------------------------------------------------------
        # Get items
        # --------------------------------------------------------------------------------------------------------------
        for item in items.split(";"):
            # Get rid of space
            item = item.strip()
            # Skip
            if not item: continue
            # 1> Get name & value
            name, value = item.split(": ")
            # 2> Save
            result.update({name: value})
        # Return
        return result

    @staticmethod
    def parse_logger(*args, **kwargs):
        """
        :param args:
            - log_dir, log_name or
            - log_path
        :return: List of Dict of
            {
                - time:
                - titles:
                - counters:
                - item_name1: item_value1;
                - item_name2: item_value2:
                ...
            }
        """
        # 1. Find logger
        if len(args) == 1:
            logger = args[0]
        else:
            log_dir, log_name = args
            # Search
            logger = []
            for obj in os.listdir(log_dir):
                if obj.endswith("_%s.log" % log_name):
                    logger.append(os.path.join(log_dir, obj))
            assert len(logger) == 1, "Too many loggers satisfy the given log name: \n\t%s. " % logger
            logger = logger[0]
        # 2. Parse logger
        # (1) Init result
        r = []
        # (2) Parse each line
        # 1> Open file
        f = open(logger, 'r')
        # 2> Parsing
        lines = f.readlines()
        for index, current_line in enumerate(lines):
            # Show progress
            if chk_d(kwargs, 'verbose'):
                show_progress(title="Parsing log file '%s'" % os.path.split(logger)[1], index=index, maximum=len(lines))
            # 1. Try to parse current line
            try:
                current_result = Logger.parse_logger_line(current_line)
                r.append(current_result)
            # 2. Abort
            except:
                continue
        # 3> Close file
        f.close()
        # Return
        return r

    @staticmethod
    def reform_no_display_items(items):
        """
        For no-display item: key should contain "-NO_DISPLAY".
        :param items:
        :return:
        """
        # Init result
        ret, no_display_keys = {}, []
        # Process
        for key, value in items.items():
            if "_NO_DISPLAY" in key:
                key = key.replace("_NO_DISPLAY", "")
                no_display_keys.append(key)
            ret[key] = value
        # Return
        return {'items': ret, "no_display_keys": no_display_keys}


########################################################################################################################
# TFboard
########################################################################################################################

def tfboard_add_multi_scalars(tfboard, multi_scalars, global_step):
    for main_tag, main_tag_value in multi_scalars.items():
        tfboard.add_scalars(main_tag, main_tag_value, global_step=global_step)
    tfboard.flush()


########################################################################################################################
# Show information
########################################################################################################################

if sys.version_info < (3, 0):
    from .show_progress_py2 import show_progress_py2 as show_progress
else:
    from .show_progress_py3 import show_progress_py3 as show_progress


def show_arguments(args, title=None, exclude_keys=None,
                   default_args=None, compared_args=None, competitor_name='default'):
    """
    Display args:
        -------------------------------------
        TITLE
        -------------------------------------
        arg1:\t\t\t value1 \t\t\t [default: value, competitor: value]
        arg2:\t\t\t value2 \t\t\t [competitor: value]
        arg3:\t\t\t value3 \t\t\t [default: value]
        -------------------------------------
    :param args: Typically induced from parsers.arg_parse()
    :param title:
    :param exclude_keys:
    :param default_args:
    :param compared_args:
    :param competitor_name:
    :return: Echo str.
    """
    # Init result
    echo_str = '\n'
    # 1. Convert to dict.
    if not isinstance(args, dict): args = vars(args)
    if default_args is not None and not isinstance(default_args, dict): default_args = vars(default_args)
    if compared_args is not None and not isinstance(compared_args, dict): compared_args = vars(compared_args)
    # 2. Get keys in an order.
    keys = sorted(list(args.keys()))
    # 3. Show information as following format:
    #   ---------- (*100)
    #   arg1:\t\t\t value1 \t\t\t [default: value]
    #   arg2:\t\t\t value2 \t\t\t [default: value]
    #   ---------- (*100)
    # (1) Head line
    echo_str += ('=' * 125 + '\n')
    # (2) Title
    if title is not None:
        echo_str += '\t%s\n' % title
        echo_str += ('-' * 125 + '\n')
    # (3) Content
    for key in keys:
        if exclude_keys is not None and key in exclude_keys: continue
        # 1. Add original
        echo_str += '%-50s: %-30s' % (key, args[key])
        # 2. Add default & competitor
        if default_args is not None and key in default_args.keys():
            # (1) Default
            echo_str += '[default: %s' % default_args[key]
            # (2) Competitor
            if compared_args is not None and key in compared_args.keys() and compared_args[key] != args[key]:
                echo_str += ', %s: %s]\n' % (competitor_name, compared_args[key])
            else:
                echo_str += ']\n'
        else:
            if compared_args is not None and key in compared_args.keys() and compared_args[key] != args[key]:
                echo_str += '[%s: %s]\n' % (competitor_name, compared_args[key])
            else:
                echo_str += '\n'
    # (4) Tail line
    echo_str += '=' * 125
    # Return
    return echo_str
