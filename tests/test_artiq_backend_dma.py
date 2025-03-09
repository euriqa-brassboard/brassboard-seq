#

import dummy_artiq
dummy_artiq.inject()

from check_artiq_backend import *
rtio_mgr.set_use_dma()
