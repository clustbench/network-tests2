all: $(patsubst %.ts, %.qm, $(wildcard  *.ts))

%.qm: %.ts
	lrelease $^ -qm $@

clean:
	rm -f *.qm

