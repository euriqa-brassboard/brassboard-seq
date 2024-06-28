#

import types
import sys
import re

def new_module(parent, name):
    m = types.ModuleType(name)
    if parent is not None:
        setattr(parent, name, m)
    return m

artiq = new_module(None, 'artiq')

language = new_module(artiq, 'language')
environment = new_module(language, 'environment')

coredevice = new_module(artiq, 'coredevice')

ad9910 = new_module(coredevice, 'ad9910')
ad9910._AD9910_REG_CFR1 = 0x00
ad9910._AD9910_REG_CFR2 = 0x01
ad9910._AD9910_REG_CFR3 = 0x02
ad9910._AD9910_REG_AUX_DAC = 0x03
ad9910._AD9910_REG_IO_UPDATE = 0x04
ad9910._AD9910_REG_FTW = 0x07
ad9910._AD9910_REG_POW = 0x08
ad9910._AD9910_REG_ASF = 0x09
ad9910._AD9910_REG_SYNC = 0x0a
ad9910._AD9910_REG_RAMP_LIMIT = 0x0b
ad9910._AD9910_REG_RAMP_STEP = 0x0c
ad9910._AD9910_REG_RAMP_RATE = 0x0d
ad9910._AD9910_REG_PROFILE0 = 0x0e
ad9910._AD9910_REG_PROFILE1 = 0x0f
ad9910._AD9910_REG_PROFILE2 = 0x10
ad9910._AD9910_REG_PROFILE3 = 0x11
ad9910._AD9910_REG_PROFILE4 = 0x12
ad9910._AD9910_REG_PROFILE5 = 0x13
ad9910._AD9910_REG_PROFILE6 = 0x14
ad9910._AD9910_REG_PROFILE7 = 0x15
ad9910._AD9910_REG_RAM = 0x16

edge_counter = new_module(coredevice, 'edge_counter')
edge_counter.CONFIG_COUNT_RISING = 0b0001
edge_counter.CONFIG_COUNT_FALLING = 0b0010
edge_counter.CONFIG_SEND_COUNT_EVENT = 0b0100
edge_counter.CONFIG_RESET_TO_ZERO = 0b1000

spi2 = new_module(coredevice, 'spi2')
spi2.SPI_DATA_ADDR = 0
spi2.SPI_CONFIG_ADDR = 1
spi2.SPI_OFFLINE = 0x01
spi2.SPI_END = 0x02
spi2.SPI_INPUT = 0x04
spi2.SPI_CS_POLARITY = 0x08
spi2.SPI_CLK_POLARITY = 0x10
spi2.SPI_CLK_PHASE = 0x20
spi2.SPI_LSB_FIRST = 0x40
spi2.SPI_HALF_DUPLEX = 0x80

ttl = new_module(coredevice, 'ttl')

urukul = new_module(coredevice, 'urukul')
urukul.SPI_CONFIG = spi2.SPI_CS_POLARITY
urukul.SPIT_CFG_WR = 2
urukul.SPIT_CFG_RD = 16
urukul.SPIT_ATT_WR = 6
urukul.SPIT_ATT_RD = 16
urukul.SPIT_DDS_WR = 2
urukul.SPIT_DDS_RD = 16

class SPIBus:
    def __init__(self, channel):
        self.ref_period_mu = 8
        self.channel = channel

class CPLD:
    def __init__(self, io_update):
        self.io_update = io_update

class AD9910:
    def __init__(self, sw, bus, cpld, chip_select):
        self.ftw_per_hz = 4.294967296
        self.sw = sw
        self.bus = bus
        self.cpld = cpld
        self.chip_select = chip_select

class EdgeCounter:
    def __init__(self, channel):
        self.channel = channel

class TTLOut:
    def __init__(self, channel):
        self.channel = channel
        self.target_o = channel << 8

class DummyDevice:
    pass

class HasEnvironment:
    def __init__(self):
        self.dummy_chn_counter = 0
        # map from urukul board number to (SPIBus, cpld)
        self.dummy_busses = {}
        self.__devices = {}

    def __next_id(self):
        self.dummy_chn_counter += 1
        return self.dummy_chn_counter

    def __new_device(self, name):
        um = re.match('urukul(\\d+)_ch(\\d+)', name)
        if um is not None:
            urukul_num = int(um[1])
            cs = int(um[2])
            bus_cpld = self.dummy_busses.get(urukul_num)
            if bus_cpld is None:
                bus = SPIBus(self.__next_id())
                cpld = CPLD(TTLOut(self.__next_id()))
                self.dummy_busses[urukul_num] = bus, cpld
            else:
                bus, cpld = bus_cpld
            sw = TTLOut(self.__next_id())
            return AD9910(sw, bus, cpld, cs)
        elif name.startswith('ttl'):
            if name.endswith('_counter'):
                return EdgeCounter(self.__next_id())
            return TTLOut(self.__next_id())
        elif name == "dummy_dev":
            return DummyDevice()
        else:
            raise KeyError(f"Cannot find device {name}")

    def get_device(self, name):
        dev = self.__devices.get(name)
        if dev is not None:
            return dev
        dev = self.__new_device(name)
        self.__devices[name] = dev
        return dev

class DummyRegistry:
    def get_unique_device_key(self, name):
        return name

class DummyDaxSystem(HasEnvironment):
    def __init__(self):
        super().__init__()
        self.registry = DummyRegistry()

    def get_device(self, name):
        raise RuntimeError("Must not call")

ad9910.AD9910 = AD9910
edge_counter.EdgeCounter = EdgeCounter
ttl.TTLOut = TTLOut
environment.HasEnvironment = HasEnvironment

def inject():
    sys.modules['artiq'] = artiq
    sys.modules['artiq.language'] = language
    sys.modules['artiq.language.environment'] = environment
    sys.modules['artiq.coredevice'] = coredevice
    sys.modules['artiq.coredevice.ad9910'] = ad9910
    sys.modules['artiq.coredevice.edge_counter'] = edge_counter
    sys.modules['artiq.coredevice.spi2'] = spi2
    sys.modules['artiq.coredevice.ttl'] = ttl
    sys.modules['artiq.coredevice.urukul'] = urukul
