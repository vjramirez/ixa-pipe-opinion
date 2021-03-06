package eus.ixa.ixa.pipe.opinion;

import ixa.kaflib.KAFDocument;

public interface Annotate {
  
  /**
   * Extracts aspects from a text and creates an opinion layer in NAF.
   * @param kaf the NAF document
   */
  public void annotate(KAFDocument kaf);
  
  /**
   * Serializes the NAF containing opinion layer with aspects.
   * @param kaf the NAF document
   * @return the NAF document in a string
   */
  public String annotateToNAF(KAFDocument kaf);

}
