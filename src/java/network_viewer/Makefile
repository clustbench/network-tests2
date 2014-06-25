include ../../../config

#ANT=ant

all: network_viewer

network_viewer:
	$(ANT) dist

install: network_viewer
	cp -rf ./network_viewer.jar $(INSTALL_DIR)/java/
	echo "#!/bin/sh" > network_viewer
	echo >> network_viewer
	echo >> network_viewer
	echo "# -Xmx1024m  is an extention of the default JVM limit to 1Gb." >> network_viewer
	echo "#  The default value is 128mb." >> network_viewer
	echo -n "java -Xmx1024m -jar $(INSTALL_DIR)/java/network_viewer.jar " >> network_viewer
	echo '"$$@"' >> network_viewer
	chmod 0755 network_viewer
	cp -rf ./network_viewer $(INSTALL_DIR)/bin/network_viewer 

clean:
	$(ANT) clean
	rm -f network_viewer.jar network_viewer

.PHONY: network_viewer

