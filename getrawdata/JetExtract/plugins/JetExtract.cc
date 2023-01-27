// -*- C++ -*-
//
// Package:    test/JetExtract
// Class:      JetExtract
// 
/**\class JetExtract JetExtract.cc test/JetExtract/plugins/JetExtract.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  
//         Created:  Sat, 31 Dec 2022 13:34:53 GMT
//
//


// system include files
#include <memory>
#include <TMath.h>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include <DataFormats/TrackReco/interface/Track.h>


//classes to save data
#include "TTree.h"
#include "TFile.h"
#include<vector>
//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.

class JetExtract : public edm::one::EDAnalyzer<edm::one::SharedResources> {
   public:
      explicit JetExtract(const edm::ParameterSet&);
      ~JetExtract();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      // ----------member data ---------------------------
      edm::EDGetTokenT<pat::JetCollection> fatjetToken_;
      int numfatjet = 0; //number of jets in the event
      TTree *mtree;

      std::vector<int> jetID;
      std::vector<double> jetArea, jet_Px, jet_Py, jet_Pz, jet_E, jet_phi, jet_theta, jet_vx, jet_vy, jet_vz,
                          jet_polarPx, jet_polarPy, jet_polarPz, jet_polarE, jet_rapidity;
      std::vector<int> particleID;
      std::vector<double> particle_Px, particle_Py, particle_Pz, particle_E, particle_phi, particle_theta, particle_vx, particle_vy, particle_vz,
                          particle_polarPx, particle_polarPy, particle_polarPz, particle_polarE, particle_rapidity;
                          
      std::vector<bool> isMuon, isElec, isPhoton, isJet;
      //std::vector<const reco::Track*> bestTracks;   //need a dictionary
      
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
JetExtract::JetExtract(const edm::ParameterSet& iConfig):
   fatjetToken_(consumes<pat::JetCollection>(iConfig.getParameter<edm::InputTag>("fatjets")))
{
   //now do what ever initialization is needed
   edm::Service<TFileService> fs;
   mtree = fs->make<TTree>("Events", "Events");

   mtree->Branch("numberfatjet",&numfatjet);
   mtree->GetBranch("numberfatjet")->SetTitle("Number of Fatjets in the Event");
   mtree->Branch("jetID",&jetID);
   mtree->GetBranch("jetID")->SetTitle("The jet in the Event to which the particle belongs");
   mtree->Branch("jetArea",&jetArea);
   mtree->GetBranch("jetArea")->SetTitle("Jet area");
   mtree->Branch("jet_vx",&jet_vx);
   mtree->GetBranch("jet_vx")->SetTitle("jet x coordinate of vertix");
   mtree->Branch("jet_vy",&jet_vy);
   mtree->GetBranch("jet_vy")->SetTitle("jet y coordinate of vertix");
   mtree->Branch("jet_vz",&jet_vz);
   mtree->GetBranch("jet_vz")->SetTitle("jet z coordinate of vertix");
   mtree->Branch("jet_Px",&jet_Px);
   mtree->GetBranch("jet_Px")->SetTitle("jet x coordinate of 4-momentum");
   mtree->Branch("jet_Py",&jet_Py);
   mtree->GetBranch("jet_Py")->SetTitle("jet y coordinate of 4-momentum");
   mtree->Branch("jet_Pz",&jet_Pz);
   mtree->GetBranch("jet_Pz")->SetTitle("jet z coordinate of 4-momentum");
   mtree->Branch("jet_E",&jet_E);
   mtree->GetBranch("jet_E")->SetTitle("jet Energy coordinate of 4-momentum");
   mtree->Branch("jet_polarPx",&jet_polarPx);
   mtree->GetBranch("jet_polarPx")->SetTitle("jet x coordinate of polar 4-momentum, i.e. transverse momentum");
   mtree->Branch("jet_polarPy",&jet_polarPy);
   mtree->GetBranch("jet_polarPy")->SetTitle("jet y coordinate of polar 4-momentum, i.e. pseudorapidity");
   mtree->Branch("jet_polarPz",&jet_polarPz);
   mtree->GetBranch("jet_polarPz")->SetTitle("jet z coordinate of polar 4-momentum, i.e. phi");
   mtree->Branch("jet_polarE",&jet_polarE);
   mtree->GetBranch("jet_polarE")->SetTitle("jet Energy coordinate of polar 4-momentum, i.e. mass");
   mtree->Branch("jet_rapidity",&jet_rapidity);
   mtree->GetBranch("jet_rapidity")->SetTitle("jet rapidity");
   mtree->Branch("jet_phi",&jet_phi);
   mtree->GetBranch("jet_phi")->SetTitle("jet azimuthal angle");
   mtree->Branch("jet_theta",&jet_theta);
   mtree->GetBranch("jet_theta")->SetTitle("jet polar angle");

   mtree->Branch("particleType",&particleID);
   mtree->Branch("particle_vx",&particle_vx);
   mtree->GetBranch("particle_vx")->SetTitle("particle x coordinate of vertix");
   mtree->Branch("particle_vy",&particle_vy);
   mtree->GetBranch("particle_vy")->SetTitle("particle y coordinate of vertix");
   mtree->Branch("particle_vz",&particle_vz);
   mtree->GetBranch("particle_vz")->SetTitle("particle z coordinate of vertix");
   mtree->Branch("particle_Px",&particle_Px);
   mtree->GetBranch("particle_Px")->SetTitle("particle x coordinate of 4-momentum");
   mtree->Branch("particle_Py",&particle_Py);
   mtree->GetBranch("particle_Py")->SetTitle("particle y coordinate of 4-momentum");
   mtree->Branch("particle_Pz",&particle_Pz);
   mtree->GetBranch("particle_Pz")->SetTitle("particle z coordinate of 4-momentum");
   mtree->Branch("particle_E",&particle_E);
   mtree->GetBranch("particle_E")->SetTitle("particle Energy coordinate of 4-momentum");
   mtree->Branch("particle_polarPx",&particle_polarPx);
   mtree->GetBranch("particle_polarPx")->SetTitle("particle x coordinate of polar 4-momentum, i.e. transverse momentum");
   mtree->Branch("particle_polarPy",&particle_polarPy);
   mtree->GetBranch("particle_polarPy")->SetTitle("particle y coordinate of polar 4-momentum, i.e. pseudorapidity");
   mtree->Branch("particle_polarPz",&particle_polarPz);
   mtree->GetBranch("particle_polarPz")->SetTitle("particle z coordinate of polar 4-momentum, i.e. phi");
   mtree->Branch("particle_polarE",&particle_polarE);
   mtree->GetBranch("particle_polarE")->SetTitle("particle Energy coordinate of polar 4-momentum, i.e. mass");
   mtree->Branch("particle_rapidity",&particle_rapidity);
   mtree->GetBranch("particle_rapidity")->SetTitle("particle rapidity");
   mtree->Branch("particleType",&particleID);
   mtree->GetBranch("particleType")->SetTitle("Particle type");
   mtree->Branch("particle_phi",&particle_phi);
   mtree->GetBranch("particle_phi")->SetTitle("particle azimuthal angle");
   mtree->Branch("particle_theta",&particle_theta);
   mtree->GetBranch("particle_theta")->SetTitle("particle polar angle");
   mtree->Branch("isMuon",&isMuon);
   mtree->GetBranch("isMuon")->SetTitle("is Muon");
   mtree->Branch("isElectron",&isElectron);
   mtree->GetBranch("isElectron")->SetTitle("is electron");
   mtree->Branch("isPhoton",&isPhoton);
   mtree->GetBranch("isPhoton")->SetTitle("is photon");
   //mtree->Branch("bestTracks",&bestTracks, 2);
   //mtree->GetBranch("bestTracks")->SetTitle("best Track pointer");

}


JetExtract::~JetExtract()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
JetExtract::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   Handle<pat::JetCollection> fatjets;
   iEvent.getByToken(fatjetToken_, fatjets);

   jetID.clear();
   particleID.clear();
   particle_vx.clear();
   particle_vy.clear();
   particle_vz.clear();
   particle_Px.clear();
   particle_Py.clear();
   particle_Pz.clear();
   particle_E.clear();
   particle_polarPx.clear();
   particle_polarPy.clear();
   particle_polarPz.clear();
   particle_polarE.clear();
   particle_phi.clear();
   particle_theta.clear();
   particle_rapidity.clear();
   isMuon.clear();
   isElec.clear();
   isPhoton.clear();
   isJet.clear();
   //bestTracks.clear();
   jetArea.clear();
   jet_vx.clear();
   jet_vy.clear();
   jet_vz.clear();
   jet_Px.clear();
   jet_Py.clear();
   jet_Pz.clear();
   jet_E.clear();
   jet_polarPx.clear();
   jet_polarPy.clear();
   jet_polarPz.clear();
   jet_polarE.clear();
   jet_phi.clear();
   jet_theta.clear();
   jet_rapidity.clear();

   if(fatjets.isValid()){

      for (const pat::Jet &jet : *fatjets){

         cout << "numfatjet " << numfatjet << endl;
         cout << "is PFJet? " << jet.isPFJet() << endl;
         cout << "is CaloJet? " << jet.isCaloJet() << endl;
         
         jet_vx.push_back(jet.vx());
         jet_vy.push_back(jet.vy());
         jet_vz.push_back(jet.vz());
         jet_theta.push_back(jet.theta());
         jet_phi.push_back(jet.phi());
         jet_Px.push_back(jet.p4().Px());              // Lorentz vector with cylindrical internal representation using pseudorapidity.
         jet_Py.push_back(jet.p4().Py());
         jet_Pz.push_back(jet.p4().Pz());
         jet_E.push_back(jet.p4().E());
         jet_polarPx.push_back(jet.polarP4().Pt());              // Lorentz vector with cylindrical internal representation using pseudorapidity.
         jet_polarPy.push_back(jet.polarP4().eta());
         jet_polarPz.push_back(jet.polarP4().phi());
         jet_polarE.push_back(jet.polarP4().M());
         jet_rapidity.push_back(jet.rapidity());
         jetArea.push_back(jet.jetArea());
         
         for (const reco::Jet::Constituent &particlePtr : jet.getJetConstituents()) {
            if (! particlePtr) {
               cout << "non valid ptr, skipping..." << endl;
               continue;
            }

            //cout << "particle id: " << particlePtr->pdgId() << endl;
            jetID.push_back(numfatjet);
            particleID.push_back(particlePtr->pdgId());
            particle_vx.push_back(particlePtr->vx());
            particle_vy.push_back(particlePtr->vy());
            particle_vz.push_back(particlePtr->vz());
            particle_theta.push_back(particlePtr->theta());
            particle_phi.push_back(particlePtr->phi());
            particle_Px.push_back(particlePtr->p4().Px());              // Lorentz vector with cylindrical internal representation using pseudorapidity.
            particle_Py.push_back(particlePtr->p4().Py());
            particle_Pz.push_back(particlePtr->p4().Pz());
            particle_E.push_back(particlePtr->p4().E());
            particle_polarPx.push_back(particlePtr->polarP4().Pt());              // Lorentz vector with cylindrical internal representation using pseudorapidity.
            particle_polarPy.push_back(particlePtr->polarP4().eta());
            particle_polarPz.push_back(particlePtr->polarP4().phi());
            particle_polarE.push_back(particlePtr->polarP4().M());
            particle_rapidity.push_back(particlePtr->rapidity());

            isMuon.push_back(particlePtr->isMuon());
            isElec.push_back(particlePtr->isElectron());
            isPhoton.push_back(particlePtr->isPhoton());
            isJet.push_back(particlePtr->isJet());
            //bestTracks.push_back(particlePtr->bestTrack());

         }

         ++numfatjet;
      }
      
      mtree->Fill();
   }


   return;

}


// ------------ method called once each job just before starting event loop  ------------
void 
JetExtract::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
JetExtract::endJob() 
{
   std::cout << "Total number of processed jets: " << numfatjet << std::endl;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
JetExtract::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetExtract);
