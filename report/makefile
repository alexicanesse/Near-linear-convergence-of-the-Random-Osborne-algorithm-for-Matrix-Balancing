TIME     = /usr/bin/time -p
LATEXMK  = latexmk -silent -f -g --pdf
LUALATEX = lualatex -interaction=batchmode --shell-escape
XELATEX  = xelatex -interaction=batchmode --shell-escape
PDFCROP  = pdfcrop
RM       = /bin/rm

default : rapport.pdf

StandAloneGraphicsTeXFiles = $(wildcard figures/*_sag.tex)

PDFGraphics = $(patsubst %_sag.tex,%_sag.pdf,$(StandAloneGraphicsTeXFiles))

InputTeXFiles = $(wildcard *_input.tex)

%_sag.pdf : %_sag.tex
	$(TIME) $(LUALATEX) --output-directory=figures $< || true
	#$(PDFCROP) $@ $@

noref : $(PDFGraphics) $(InputTeXFiles) rapport.tex
	$(TIME) $(LUALATEX) rapport || true

rapport.pdf : $(PDFGraphics) $(InputTeXFiles) rapport.tex
	$(TIME) $(LUALATEX) rapport; $(TIME) biber rapport; $(TIME) $(LUALATEX) rapport || true

pres.pdf : $(PDFGraphics) $(InputTeXFiles) pres.tex
	$(TIME) $(LUALATEX) pres.tex; $(TIME) biber pres; $(TIME) $(LUALATEX) pres.tex || true

clean : .PHONY
	$(RM) -f -- *.aux *.bak *.bbl *.blg *.log *.out *.toc *.tdo _region.* *.bcf *.run.xml *.upa *.upb *.fls *.auxlock

depclean : clean
	$(RM) -f -- *_sag.pdf

distclean : depclean
	$(RM) -f -- Project.pdf

.PHONY :
