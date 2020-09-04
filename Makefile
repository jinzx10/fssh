ROOTDIR 		= $(shell pwd)
BDIR 			= ${ROOTDIR}/bin
IDIR 			= ${ROOTDIR}/include
ODIR 			= ${ROOTDIR}/obj
SDIR 			= ${ROOTDIR}/src

CC 				= mpicxx
CPPFLAGS 		= -std=c++11 -O2 -I${IDIR}
LDFLAGS 		= 

vpath %.h ${IDIR}
vpath %.cpp ${SDIR}

% 				: ${BDIR}/% ;

${BDIR}/run_tully1 		: ${ODIR}/run_tully1.o ${ODIR}/tully.o ${ODIR}/fssh1990.o | ${BDIR}
	${CC} $^ -o $@ ${CPPFLAGS} ${LDFLAGS}

${ODIR}/run_tully1.o 	: run_tully1.cpp | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}

${ODIR}/%.o				: %.cpp %.h | ${ODIR}
	${CC} -c $< -o $@ ${CPPFLAGS}

${BDIR} ${ODIR} 		:
	mkdir -p $@


.PRECIOUS: ${ODIR}/%.o
.PHONY: clean

clean:
	rm -rf ${ODIR} ${BDIR} ${LDIR}
