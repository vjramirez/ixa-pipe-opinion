/*
 *  Copyright 2015 Rodrigo Agerri

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

package eus.ixa.ixa.pipe.opinion;

import ixa.kaflib.KAFDocument;
import ixa.kaflib.Opinion;
import ixa.kaflib.Term;
import ixa.kaflib.Term.Sentiment;
import ixa.kaflib.WF;
import ixa.kaflib.Opinion.OpinionExpression;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import com.google.common.io.Files;

import eus.ixa.ixa.pipe.ml.StatisticalSequenceLabeler;
import eus.ixa.ixa.pipe.ml.polarity.DictionaryPolarityTagger;
import eus.ixa.ixa.pipe.ml.sequence.SequenceLabel;
import eus.ixa.ixa.pipe.ml.sequence.SequenceLabelFactory;
import eus.ixa.ixa.pipe.ml.sequence.SequenceLabelerME;
import eus.ixa.ixa.pipe.ml.sequence.SequenceLabelSample;
import eus.ixa.ixa.pipe.ml.utils.Span;

/**
 * Annotation class for Opinion Target Extraction (OTE).
 * 
 * @author ragerri
 * @version 2015-04-29
 * 
 */
public class Annotate {

  /**
   * The factory to construct Name objects.
   */
  private SequenceLabelFactory nameFactory;
  /**
   * The NameFinder to do the opinion target extraction.
   */
  private StatisticalSequenceLabeler oteExtractor;
  /**
   * Clear features after every sentence or when a -DOCSTART- mark appears.
   */
  private String clearFeatures;
  /**
   * Path to the lexicon used for polarity.
   */
  private String lexicon;

  
  public Annotate(final Properties properties) throws IOException {

    if (properties.getProperty("lexicon") != null ) {
    	this.lexicon = properties.getProperty("lexicon");
    }
    else {
        this.clearFeatures = properties.getProperty("clearFeatures");
        nameFactory = new SequenceLabelFactory();
        oteExtractor = new StatisticalSequenceLabeler(properties, nameFactory);
    }
  }
  
  /**
   * Extract Opinion Targets.
   * @param kaf the KAFDocument
   * @throws IOException if io errors
   */
  public final void annotateOTE(final KAFDocument kaf) throws IOException {

    List<List<WF>> sentences = kaf.getSentences();
    for (List<WF> sentence : sentences) {
      //process each sentence
      String[] tokens = new String[sentence.size()];
      String[] tokenIds = new String[sentence.size()];
      for (int i = 0; i < sentence.size(); i++) {
        tokens[i] = sentence.get(i).getForm();
        tokenIds[i] = sentence.get(i).getId();
      }
      if (clearFeatures.equalsIgnoreCase("docstart") && tokens[0].startsWith("-DOCSTART-")) {
        oteExtractor.clearAdaptiveData();
      }
      List<SequenceLabel> names = oteExtractor.getSequences(tokens);
      for (SequenceLabel name : names) {
        Integer startIndex = name.getSpan().getStart();
        Integer endIndex = name.getSpan().getEnd();
        List<Term> nameTerms = kaf.getTermsFromWFs(Arrays.asList(Arrays
            .copyOfRange(tokenIds, startIndex, endIndex)));
        ixa.kaflib.Span<Term> oteSpan = KAFDocument.newTermSpan(nameTerms);
        Opinion opinion = kaf.newOpinion();
        opinion.createOpinionTarget(oteSpan);
        //TODO expression span, perhaps heuristic around ote?
        OpinionExpression opExpression = opinion.createOpinionExpression(oteSpan);
        opExpression.setSentimentProductFeature(name.getType());
      }
      if (clearFeatures.equalsIgnoreCase("yes")) {
        oteExtractor.clearAdaptiveData();
      }
    }
    oteExtractor.clearAdaptiveData();
  }

  /**
   * Extract Polarity.
   * @param kaf the KAFDocument
   * @throws IOException if io errors
   */
  public final void annotatePOL(final KAFDocument kaf) throws IOException {
	DictionaryPolarityTagger dict = new DictionaryPolarityTagger(lexicon);
	List<Term> terms = kaf.getTerms();
	for(Term term: terms) {
		String lemma = term.getLemma();
		String text = term.getStr();
		String polarity=dict.apply(lemma);
		if (polarity == "O") {
			polarity=dict.apply(text);
		}
		if (polarity != "O") {
			Sentiment sentiment = term.createSentiment();
			sentiment.setPolarity(polarity);
			sentiment.setResource(Files.getNameWithoutExtension(lexicon));
		}
		
	}
  }
  
  
  /**
   * Output annotation as NAF.
   * 
   * @param kaf
   *          the naf document
   * @return the string containing the naf document
   */
  public final String annotateOTEsToKAF(KAFDocument kaf) {
    return kaf.toString();
  }
  
  /**
   * Output annotation in OpenNLP format.
   * 
   * @param kaf
   *          the naf document
   * @return the string containing the annotated document
   */
  public final String annotateOTEsToOpenNLP(KAFDocument kaf) {
    StringBuilder sb = new StringBuilder();
    List<List<WF>> sentences = kaf.getSentences();
    for (List<WF> sentence : sentences) {
      String[] tokens = new String[sentence.size()];
      String[] tokenIds = new String[sentence.size()];
      for (int i = 0; i < sentence.size(); i++) {
        tokens[i] = sentence.get(i).getForm();
        tokenIds[i] = sentence.get(i).getId();
      }
      if (clearFeatures.equalsIgnoreCase("docstart") && tokens[0].startsWith("-DOCSTART-")) {
        oteExtractor.clearAdaptiveData();
      }
      Span[] statSpans = oteExtractor.seqToSpans(tokens);
      boolean isClearAdaptiveData = false;
      if (clearFeatures.equalsIgnoreCase("yes")) {
        isClearAdaptiveData = true;
      }
      Span[] allSpansArray = SequenceLabelerME.dropOverlappingSpans(statSpans);
      SequenceLabelSample nameSample = new SequenceLabelSample(tokens, allSpansArray, isClearAdaptiveData);
      sb.append(nameSample.toString()).append("\n");
    }
    oteExtractor.clearAdaptiveData();
    return sb.toString();
  }

}
