# Use 1.3 because 1.4 has a ton of other things that I think we don't need for the estimate...
FROM jetscape/base:v1.3

COPY --chown=jetscape-user:jetscape-user JETSCAPE /home/jetscape-user/JETSCAPE
#RUN git clone https://github.com/JETSCAPE/JETSCAPE.git \
RUN cd JETSCAPE/external_packages \
        && ./get_lbtTab.sh \
        && cd ../examples \
        && ./get_hydroSample_PbPb2760_cen_00-05.sh \
        && cd .. && mkdir build && cd build \
        && cmake .. && make

COPY --chown=jetscape-user:jetscape-user run_jetscape.sh /home/jetscape-user/
RUN cd && mkdir bin && cd bin \
        && ln -s ../JETSCAPE/build/runJetscape . \
        && ln -s ../JETSCAPE/build/FinalStateHadrons . \
        && ln -s ../JETSCAPE/build/FinalStatePartons . \
        && ln -s /home/jetscape-user/run_jetscape.sh . && cd && ln -s /home/jetscape-user/JETSCAPE/build/LBT-tables .
ENV PATH="${PATH}:${HOME}/bin"

ENTRYPOINT ["run_jetscape.sh"]
