# Choose which style to pass to docswitch
# NB: Default value here is always used by Overleaf
$theflag = $ENV{'STYLEFLAG'} || '\def\flag{lsstdescnote}';

# Set environment variables to search for tex inputs
# http://tex.stackexchange.com/a/50847/121099
$ENV{'TEXINPUTS'}='./desc-tex/:./tables/:' . $ENV{'TEXINPUTS'};
$ENV{'BSTINPUTS'}='./desc-tex/bib/:' . $ENV{'BSTINPUTS'};

# Set control tools
$go_mode = 1;
$pdf_mode = 1;
# You can change the \flag value here...
$pdflatex = 'openout_any=a pdflatex %O -interaction=nonstopmode "' . $theflag . '\input{%S}"'
