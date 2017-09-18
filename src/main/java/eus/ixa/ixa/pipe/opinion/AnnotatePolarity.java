/*
 *  Copyright 2017 Rodrigo Agerri

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

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import com.google.common.io.Files;

import eus.ixa.ixa.pipe.ml.StatisticalDocumentClassifier;
import eus.ixa.ixa.pipe.ml.polarity.DictionaryPolarityTagger;
import eus.ixa.ixa.pipe.ml.utils.Flags;
import eus.ixa.ixa.pipe.ml.utils.StringUtils;
import ixa.kaflib.KAFDocument;
import ixa.kaflib.Opinion;
import ixa.kaflib.Opinion.OpinionExpression;
import ixa.kaflib.Span;
import ixa.kaflib.Term;
import ixa.kaflib.Term.Sentiment;
import ixa.kaflib.WF;
import opennlp.tools.util.StringUtil;

/**
 * Annotation class for polarity tagging using document classification.
 * 
 * @author ragerri
 * @version 2017-06-09
 * 
 */
public class AnnotatePolarity implements Annotate {

  /**
   * The Document classifier to extract the aspects.
   */
  private StatisticalDocumentClassifier polTagger;
  private DictionaryPolarityTagger dictTagger;
  private Boolean isDict = false;
  private String dictionary = null;
  private String windowMin = "N";
  private String windowMax = "N";
  /**
   * Clear features after every sentence or when a -DOCSTART- mark appears.
   */
  private String clearFeatures;

  
  public AnnotatePolarity(final Properties properties) throws IOException {

    this.clearFeatures = properties.getProperty("clearFeatures");
    dictionary = properties.getProperty("dictionary");
    windowMin = properties.getProperty("windowMin");
    windowMax = properties.getProperty("windowMax");
    if (!dictionary.equalsIgnoreCase(Flags.DEFAULT_DICT_OPTION)) {
      dictTagger = new DictionaryPolarityTagger(new FileInputStream(dictionary));
      isDict = true;
    }
    polTagger = new StatisticalDocumentClassifier(properties);
  }
  
  /**
   * Annotate polarity using a document classifier.
   * @param kaf the KAFDocument
   */
  public final void annotate(final KAFDocument kaf) {

    List<List<WF>> sentences = kaf.getSentences();
    List<Term> terms = kaf.getTerms();
    
    //Flag to avoid creating new opinion tags if naf contains at least one.
    boolean flagOpinions = false;
    if(kaf.getOpinions().size()>0) flagOpinions=true;
    
    for (List<WF> sentence : sentences) {
      //process each sentence
      String[] tokens = new String[sentence.size()];
      String[] tokenIds = new String[sentence.size()];
      for (int i = 0; i < sentence.size(); i++) {
        tokens[i] = sentence.get(i).getForm();
        tokenIds[i] = sentence.get(i).getId();
      }
      if (isDict) {
        for (Term term : terms) {
          String polarity = dictTagger.tag(term.getForm());
          if (polarity.equalsIgnoreCase("O")) {
            polarity = dictTagger.tag(term.getLemma());
          }
          if (!polarity.equalsIgnoreCase("O")) {
            Sentiment sentiment = term.createSentiment();
            sentiment.setPolarity(polarity);
            sentiment.setResource(Files.getNameWithoutExtension(dictionary));
          }
        }
      }
      if (clearFeatures.equalsIgnoreCase("docstart") && tokens[0].startsWith("-DOCSTART-")) {
        polTagger.clearFeatureData();
      }
      //Document Classification
      Integer sentNumber = sentence.get(0).getSent();
      List<Opinion> opinionsBySentence = getOpinionsBySentence(kaf,
              sentNumber);
      
     //if exists opinion tags, the polarity is assigned for each one.
      if(opinionsBySentence.size()>0) {
    	  for (Opinion opinion : opinionsBySentence) {
    		  OpinionExpression opExpression = opinion.getOpinionExpression();
    		  String polarity = "";
    		  
    		  //If the sentence has just one opinion tag, use all tokens if not
    		  if (opinionsBySentence.size()>1) {
    			  
                  List<Term> opinionTerms = opExpression.getTerms();
                  
                  //String minTermId = opinionTerms.get(0).getId();
                  String minWFId = opinionTerms.get(0).getWFs().get(0).getId();
                  String maxWFId = opinionTerms.get(opinionTerms.size()-1).getWFs().get(0).getId();
                  
                  
        		  int minWindow = 5000;
        		  int maxWindow = 5000;
        		  if (!windowMin.equalsIgnoreCase("N")) {
                	  minWindow = Integer.parseInt(windowMin);
                  } 
        		  if (!windowMax.equalsIgnoreCase("N")) {
                	  maxWindow = Integer.parseInt(windowMax);
                  } 
        		  int minIndex = -1;
        		  int maxIndex = -1;
        		  
        		  
        		  //Getting max and min indexes for the target.
        		  List<String> tokenWindow = new ArrayList<>();
        		  for (int i = 0; i < tokenIds.length; i++) {
    				if (tokenIds[i].equals(minWFId)) {
    					minIndex=i;
    				}
    				if (tokenIds[i].equals(maxWFId)) {
    					maxIndex=i;
    				}
        		  }
        		  
        		  //Calculating min and max indexes for the windows around the target.
        		  minIndex = minIndex-minWindow;
        		  if (minIndex<0) minIndex=0;
        		  maxIndex = maxIndex+maxWindow;
        		  if (maxIndex>tokens.length-1) maxIndex = tokens.length-1;
        		  
        		  //Creating list of tokens for the defined window.
        		  for (int i = minIndex; i <= maxIndex; i++) {
        			  tokenWindow.add(tokens[i]);
        		  }
        		  
        		  polarity = polTagger.classify(tokenWindow.toArray(new String[0]));
                  //System.err.println(polarity + "\t" + String.join(" ", tokenWindow.toArray(new String[0])));
    		  } else {
    			  polarity = polTagger.classify(tokens);
    			  //System.err.println(polarity + "\t" + String.join(" ", tokens));
    		  }
    		  opExpression.setPolarity(polarity);
              
    	  }
      }
      else if(!flagOpinions){
    	  String polarity = polTagger.classify(tokens);
          List<Term> polarityTerms = kaf.getTermsFromWFs(Arrays.asList(Arrays
                .copyOfRange(tokenIds, 0, tokens.length)));
          ixa.kaflib.Span<Term> polaritySpan = KAFDocument.newTermSpan(polarityTerms);
          Opinion opinion = kaf.newOpinion();
          //TODO expression span, perhaps heuristic around ote and/or around opinion expression?
          OpinionExpression opExpression = opinion.createOpinionExpression(polaritySpan);
          opExpression.setPolarity(polarity);
      }
      
      if (clearFeatures.equalsIgnoreCase("yes")) {
        polTagger.clearFeatureData();
      }
    }
    polTagger.clearFeatureData();
  }
  
  private static List<Opinion> getOpinionsBySentence(KAFDocument kaf,
	      Integer sentNumber) {
	    List<Opinion> opinionList = kaf.getOpinions();
	    List<Opinion> opinionsBySentence = new ArrayList<>();
	    for (Opinion opinion : opinionList) {
	      try {
	    	  if (sentNumber.equals(
	    	          opinion.getOpinionTarget().getSpan().getFirstTarget().getSent())) {
	    	        opinionsBySentence.add(opinion);
	    	      }
			} catch (Exception e) {
				// TODO: handle exception
			}
	      
	    }
	    return opinionsBySentence;
	  }

  /**
   * Output annotation as NAF.
   * 
   * @param kaf
   *          the naf document
   * @return the string containing the naf document
   */
  public final String annotateToNAF(KAFDocument kaf) {
    return kaf.toString();
  }
  
  public final String annotatePolarityToTabulated(KAFDocument kaf) {
    return kaf.toString();
  }

}
