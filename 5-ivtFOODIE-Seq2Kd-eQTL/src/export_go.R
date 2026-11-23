args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(org.Hs.eg.db))
suppressPackageStartupMessages(library(AnnotationDbi))
ann <- AnnotationDbi::select(org.Hs.eg.db, keys=keys(org.Hs.eg.db, keytype="SYMBOL"),
                             keytype="SYMBOL", columns="GOALL")
write.csv(unique(ann[,c("SYMBOL", "GOALL")]), args[1], row.names=FALSE)
writeLines(c(paste("org.Hs.eg.db", packageVersion("org.Hs.eg.db")), capture.output(sessionInfo())),
           paste0(args[1], ".session.txt"))
