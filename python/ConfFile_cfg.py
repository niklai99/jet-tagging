import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import FWCore.ParameterSet.Types as CfgTypes

process = cms.Process("JetExtractor")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'root://eospublic.cern.ch//eos/opendata/cms/Run2015D/SingleElectron/MINIAOD/08Jun2016-v1/10000/001A703B-B52E-E611-BA13-0025905A60B6.root'
    )
)

process.demo = cms.EDAnalyzer('JetExtract',
    fatjets = cms.InputTag("slimmedJetsAK8")
)

process.TFileService = cms.Service("TFileService", fileName=cms.string("extracted_data.root"))
process.p = cms.Path(process.demo)
