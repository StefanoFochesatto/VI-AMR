all: viamr.pdf

figures = drmax.png  herrinf.png  uerrh1.png

viamr.pdf: viamr.aux viamr.bbl
	while grep 'Rerun to get ' viamr.log ; do pdflatex -shell-escape viamr.tex ; done

viamr.aux: viamr.tex drmax.png
	pdflatex -shell-escape $<

viamr.bbl: viamr.aux
	bibtex $<

drmax.png: genfigs/glacierconv.py
	(cd genfigs/ && python3 glacierconv.py && mogrify -trim *.png)

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.lo? *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz *.thm *.dvi *.fls *.fdb_latexmk
	@rm -rf result* _minted*
	@rm -f genfigs/*.png
