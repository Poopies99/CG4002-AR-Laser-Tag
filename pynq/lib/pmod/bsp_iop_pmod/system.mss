
 PARAMETER VERSION = 2.2.0


BEGIN OS
 PARAMETER OS_NAME = standalone
 PARAMETER OS_VER = 7.3
 PARAMETER PROC_INSTANCE = iop_pmoda_mb
 PARAMETER stdin = iop_pmoda_lmb_lmb_bram_if_cntlr
 PARAMETER stdout = iop_pmoda_lmb_lmb_bram_if_cntlr
END


BEGIN PROCESSOR
 PARAMETER DRIVER_NAME = cpu
 PARAMETER DRIVER_VER = 2.12
 PARAMETER HW_INSTANCE = iop_pmoda_mb
 PARAMETER compiler_flags = -mcpu=v11.0 -mlittle-endian -mxl-soft-mul
END


BEGIN DRIVER
 PARAMETER DRIVER_NAME = generic
 PARAMETER DRIVER_VER = 2.1
 PARAMETER HW_INSTANCE = address_remap_0
END

BEGIN DRIVER
 PARAMETER DRIVER_NAME = gpio
 PARAMETER DRIVER_VER = 4.7
 PARAMETER HW_INSTANCE = iop_pmoda_gpio
END

BEGIN DRIVER
 PARAMETER DRIVER_NAME = iic
 PARAMETER DRIVER_VER = 3.7
 PARAMETER HW_INSTANCE = iop_pmoda_iic
END

BEGIN DRIVER
 PARAMETER DRIVER_NAME = intc
 PARAMETER DRIVER_VER = 3.12
 PARAMETER HW_INSTANCE = iop_pmoda_intc
END

BEGIN DRIVER
 PARAMETER DRIVER_NAME = intrgpio
 PARAMETER DRIVER_VER = 4.1
 PARAMETER HW_INSTANCE = iop_pmoda_intr
END

BEGIN DRIVER
 PARAMETER DRIVER_NAME = io_switch
 PARAMETER DRIVER_VER = 1.0
 PARAMETER HW_INSTANCE = iop_pmoda_io_switch
END

BEGIN DRIVER
 PARAMETER DRIVER_NAME = mailbox_bram
 PARAMETER DRIVER_VER = 0.1
 PARAMETER HW_INSTANCE = iop_pmoda_lmb_lmb_bram_if_cntlr
END

BEGIN DRIVER
 PARAMETER DRIVER_NAME = spi
 PARAMETER DRIVER_VER = 4.7
 PARAMETER HW_INSTANCE = iop_pmoda_spi
END

BEGIN DRIVER
 PARAMETER DRIVER_NAME = tmrctr
 PARAMETER DRIVER_VER = 4.7
 PARAMETER HW_INSTANCE = iop_pmoda_timer
END


BEGIN LIBRARY
 PARAMETER LIBRARY_NAME = pynqmb
 PARAMETER LIBRARY_VER = 1.0
 PARAMETER PROC_INSTANCE = iop_pmoda_mb
END


