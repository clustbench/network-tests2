/*
 *  This file is a part of the PARUS project.
 *  Copyright (C) 2006  Alexey N. Salnikov
 *  
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 * 
 * Alexey N. Salnikov salnikov@cmc.msu.ru
 *
 */


package parus.win;

import parus.data.Network_data;
import parus.data.Matrix;
import parus.common.Objects_loader;

import javax.swing.JDialog;
import java.awt.Container;

import java.awt.LayoutManager;
import java.awt.GridLayout;
import java.awt.GridBagLayout;
import java.awt.GridBagConstraints;
import java.awt.BorderLayout;
import java.awt.Frame;

import javax.swing.JSlider;
import javax.swing.JLabel;
import javax.swing.event.ChangeListener;
import javax.swing.event.ChangeEvent;

import javax.swing.border.EtchedBorder;

import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;

import javax.swing.JButton;

import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JCheckBoxMenuItem;

import javax.swing.ImageIcon;

import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;


import java.awt.Component;

import java.io.File;
import java.io.IOException;

import java.net.URL;

import java.util.Dictionary;
import java.util.Hashtable;
import java.util.Map;
import java.util.ArrayList;


public class Network_view_window
{
	private JDialog	viewer_frame;
	private JSlider slider;
	private JPanel panel;	

	private JTextField current_message_length;
	
	private Network_data data;
	private Network_data deviation;
	private Matrix_shower matrix_shower;

	private ArrayList<Diagram_panel> diagram_panels_list=null;

	private JCheckBoxMenuItem normolize_menu_item;
	
	private ImageIcon icon;

	private class Data_window_size_action implements ActionListener
	{
		JDialog dialog;
		JTextField text_field;

		private class Butt_acct implements ActionListener
		{
			public void actionPerformed(ActionEvent e)
			{
				String str;
				str=text_field.getText();
				Integer ival=new Integer(str);
				data.set_window_size(ival.intValue());
				dialog.setVisible(false);
			}
		}

		public void actionPerformed(ActionEvent e)
		{
			String str=new String();
			dialog=new JDialog(viewer_frame,"window size",true);

			Container cpane = dialog.getContentPane();

			text_field=new JTextField(str+data.get_window_size());
			cpane.setLayout(new GridLayout(1,2));			
			
			JButton butt=new JButton("set window size");
			butt.addActionListener(new Butt_acct());

			cpane.add(text_field);
			cpane.add(butt);
			


			dialog.setSize(200,25);
			dialog.setLocationByPlatform(true);
			dialog.pack();
			dialog.setVisible(true);
			dialog.setFocusable(true);
			dialog.repaint();
		}
		
	}
	
	private class Data_window_listener implements ActionListener
	{
		public void actionPerformed(ActionEvent e)
		{
			try
			{
				data.read_next_window();
				deviation.read_next_window();

				update_messages_slider();
				set_local_normalization(normolize_menu_item.getState());
			}
			catch(IOException ex)
			{
				System.out.println("Can't read next data window: "+ex.toString());
			}
		}
	}

	private class Matrix_shower_action implements ActionListener
	{
		 public void actionPerformed(ActionEvent e)
		 {
		 	matrix_shower.setVisible(true);
			matrix_shower.repaint();
		 }
	}

	private class Slider_listener implements ChangeListener, ActionListener
	{
		public void stateChanged(ChangeEvent e)
		{
			Integer intg=data.get_nearest_message_length(slider.getValue());
			Matrix m=data.get_matrix_by_length(intg);
			Matrix d=null;
			if(deviation!=null)
			{
				d=deviation.get_matrix_by_length(intg);
			}
			

			current_message_length.setText(intg.toString());
			matrix_shower.update(m,d);

			for(int i=0; i< diagram_panels_list.size(); i++)
			{
				Diagram_panel p=diagram_panels_list.get(i);
				p.set_current_length(intg);
				p.repaint();
			}
		}

		public void actionPerformed(ActionEvent e)
		{
			Integer old_int=data.get_nearest_message_length(slider.getValue());
			String str=current_message_length.getText();
			Integer new_int;
			if(str!=null)
			{
				try
				{
					Integer tmp_int= new Integer(str); 
					new_int=data.get_nearest_message_length(tmp_int.intValue());
				}
				catch(Exception ex)
				{
					current_message_length.setText(old_int.toString());
					return;
				}
			}
			else
			{
				current_message_length.setText(old_int.toString());
				return;
			}
			
			Matrix m=data.get_matrix_by_length(new_int);
			Matrix d=null;
			if(deviation!=null)
			{
				d=deviation.get_matrix_by_length(new_int);
			}

			current_message_length.setText(new_int.toString());
			slider.setValue(new_int.intValue());
			slider.repaint();

			matrix_shower.update(m,d);

			
			for(int i=0; i< diagram_panels_list.size(); i++)
			{
				Diagram_panel p=diagram_panels_list.get(i);
				p.set_current_length(new_int);
				p.repaint();
			}

		}

	}

	private class Pair_shower_action implements ActionListener
	{
		public void actionPerformed(ActionEvent e)
		{
			int from = matrix_shower.get_pointer_x_coord();
			int to   = matrix_shower.get_pointer_y_coord();

			Map<Integer,Double> m=data.get_fixed_pair_data(from,to);

			JDialog dialog=new JDialog(viewer_frame);
			Container p=dialog.getContentPane();
			
			Integer cur_mes_length=data.get_nearest_message_length(slider.getValue());
			Diagram_panel diagr_panel = new Diagram_panel(m,cur_mes_length);
			
			p.add(diagr_panel);

			dialog.setSize(300,200);
			/*
			 * Supported from java 1.6
			 * 
			dialog.setIconImage(icon.getImage());
			*/
			dialog.setTitle("Pair ("+from+","+to+") for all messages length");
			dialog.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
			dialog.setVisible(true);

			diagram_panels_list.add(diagr_panel);

		}
	}

	private class Normolize_listener implements ActionListener
	{
		public void actionPerformed(ActionEvent e)
		{
			set_local_normalization(normolize_menu_item.getState());
		}
	}

	private class Column_listener implements ActionListener
	{
		public void actionPerformed(ActionEvent e)
		{
			int column=matrix_shower.get_pointer_y_coord();
			Matrix m=data.create_column_matrix(column);
			Matrix d=null;
			if(deviation!=null)
			{
				d=deviation.create_column_matrix(column);
			}
			Matrix_shower shower=new Matrix_shower(viewer_frame,m,d,true);
			shower.setTitle("Column number "+column+" for all messages length");
			shower.setVisible(true);
		}
	}

	private class Row_listener implements ActionListener
	{
		public void actionPerformed(ActionEvent e)
		{
			int row=matrix_shower.get_pointer_y_coord();
			Matrix m=data.create_row_matrix(row);
			Matrix d=null;
			if(deviation!=null)
			{
				d=deviation.create_row_matrix(row);
			}

			Matrix_shower shower=new Matrix_shower(viewer_frame,m,d,true);
			shower.setTitle("Row number "+row+" for all messages length");
			shower.setVisible(true);
		}
	}
/*
	private class Zoom_action implements ActionListener
	{
		public void actionPerformed(ActionEvent e)
		{
			int fromx=matrix_shower.get_pointer_x_coord();
			int fromy=matrix_shower.get_pointer_y_coord();
			int toy=matrix_shower.get_pointer_y_coord2();
			int tox=matrix_shower.get_pointer_x_coord2();


			;
			Matrix_shower shower=new Matrix_shower(viewer_frame,m,true,fromx,fromy,tox,toy);
			shower.setTitle("Row number "+row+" for all messages length");
			shower.setVisible(true);
		}
	}
*/
	public Network_view_window(Frame parent,Network_data da,Network_data de)
	{
		data=da;
		deviation=de;
		
		if(data==null)
		{
			return;
		}
			

		viewer_frame = new JDialog(parent,"Viewer of network test results");
		
		
		JMenuBar   menu_bar       = new JMenuBar();
		JMenu      options_menu   = new JMenu("Options");
		normolize_menu_item       = new JCheckBoxMenuItem("Local normalization",true);
		JMenuItem  data_window_size_menu_item = new JMenuItem("Data window size");
		data_window_size_menu_item.addActionListener(new Data_window_size_action());


		options_menu.add(normolize_menu_item);
		options_menu.add(data_window_size_menu_item);

		menu_bar.add(options_menu);
		
		viewer_frame.setJMenuBar(menu_bar);

		normolize_menu_item.addActionListener(new Normolize_listener());



		Container pane = viewer_frame.getContentPane();
		
		/*
		 * Surface space building
		 */
		
		
		this.create_main_panel();
		
		/*
		URL path=Objects_loader.get_file("images/logotype.png");
		icon=new ImageIcon(path);
		*/
		/* supported from java 1.6
		viewer_frame.setIconImage(icon.getImage());
		*/
						
		viewer_frame.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
		viewer_frame.setSize(500,300);
		viewer_frame.setVisible(true);
		
		return;

	}

	private void create_main_panel()
	{
		EtchedBorder border=new EtchedBorder();
	
		int min=data.get_min_read_message_length();
		int max=data.get_max_read_message_length();
		//System.out.printf("min=%d, max=%d\n",min,max);
		
		slider=new JSlider(min,max,min);
		
		Dictionary<Integer,JLabel> dictionary= new Hashtable<Integer,JLabel>();
		Integer mes_len[]=data.get_messages_length();
		for(int i=0;i < mes_len.length;i++)
		{
			JLabel label= new JLabel(mes_len[i].toString());
			dictionary.put(mes_len[i],label);
		}
		


		Slider_listener slider_listener= new Slider_listener();
		slider.addChangeListener(slider_listener);
		slider.setLabelTable(dictionary);
		slider.setPaintLabels(true);
		/*
		 * To think how to draw slider 
		 * 
		slider.setMajorTickSpacing(10);
		slider.setMinorTickSpacing(1);
		slider.setPaintTicks(true);
		*/
			
		current_message_length = new JTextField(new Integer(min).toString());
		current_message_length.addActionListener(slider_listener);
		
		JButton button_show_matrix= new JButton("Show matrix");
		button_show_matrix.addActionListener(new Matrix_shower_action());

		JButton button_show_pair=new JButton("Show pair");
		button_show_pair.addActionListener(new Pair_shower_action());

		JButton button_show_row=new JButton("Show row");
		button_show_row.addActionListener(new Row_listener());

		JButton button_show_column= new JButton("Show column");
		button_show_column.addActionListener(new Column_listener());
		
		//JButton button_show_smatrix=new JButton("Show Small");
		//button_show_smatrix.addActionListener(new Row_listener());


		JButton button_set_message_length= new JButton("Set message length");
		button_set_message_length.addActionListener(slider_listener);

		JButton button_next_data_window=new JButton("Next data window");
		button_next_data_window.addActionListener(new Data_window_listener());

		
		String host_names[]=data.get_host_names();
		String str="<html>";
		for(int i=0; i< host_names.length; i++)
		{
			str+=i+" : "+host_names[i]+"<br>";
		}
		str+="</html>";
		
		JScrollPane scroll_pane=new JScrollPane(new JLabel(str));
		scroll_pane.setBorder(border);
		
		JLabel l=new JLabel(data.test_parameters_toHTML()); 
		JScrollPane parameters_pane= new JScrollPane(l);
		parameters_pane.setBorder(border);
		
	
		JPanel p = new JPanel();
		JPanel panel_show_options=new JPanel();
		JPanel panel_test_parameters=new JPanel();
		JPanel slider_panel=new JPanel();
		
		GridBagConstraints grid_constrains = new GridBagConstraints();
                grid_constrains.fill = GridBagConstraints.HORIZONTAL;
		
		
		panel_test_parameters.setLayout(new GridLayout(1,2));
		panel_test_parameters.setBorder(border);
		panel_test_parameters.add(parameters_pane,grid_constrains);
		panel_test_parameters.add(scroll_pane,grid_constrains);
		
		slider_panel.setLayout(new GridBagLayout());
		slider_panel.setBorder(border);
		
		grid_constrains.weightx = 0.2;
		grid_constrains.gridx = 0;
		grid_constrains.gridy = 1;
		slider_panel.add(current_message_length,grid_constrains);
		
		grid_constrains.weightx = 0.1;
		grid_constrains.gridx = 1;
		grid_constrains.gridy = 1;
		slider_panel.add(button_set_message_length,grid_constrains);

		grid_constrains.weightx = 0.1;
		grid_constrains.gridx = 2;
		grid_constrains.gridy = 1;
		slider_panel.add(button_next_data_window,grid_constrains);
	
		grid_constrains.weightx = 1.0;
		grid_constrains.gridwidth = 3;
		grid_constrains.gridx = 0;
		grid_constrains.gridy = 0;
		slider_panel.add(slider,grid_constrains);

		
		panel_show_options.setLayout(new GridLayout(1,4));
		panel_show_options.setBorder(border);
		panel_show_options.add(button_show_matrix);
		panel_show_options.add(button_show_row);
		panel_show_options.add(button_show_column);
		panel_show_options.add(button_show_pair);
		//panel_show_options.add(button_show_smatrix);
	
		p.setLayout(new BorderLayout());
		p.add(BorderLayout.NORTH,slider_panel);		
		p.add(BorderLayout.SOUTH,panel_show_options);
		p.add(BorderLayout.CENTER,panel_test_parameters);
		
		Container pane=viewer_frame.getContentPane();

		LayoutManager manager=viewer_frame.getLayout();
		if(panel!=null)
		{
			manager.removeLayoutComponent(panel);
			pane.remove(panel);
		}		
		
		panel=p;

		pane.add(panel);
		//viewer_frame.pack();
		pane.repaint();

		Matrix mtr_data=data.get_matrix_by_length(min);
		Matrix mtr_deviation=null;
		
		if(deviation!=null)
		{
			mtr_deviation=deviation.get_matrix_by_length(min);
		}


		if(matrix_shower==null)
		{
			if(mtr_deviation!=null)
			{
				matrix_shower=new Matrix_shower(viewer_frame,mtr_data,mtr_deviation,true);
			}
			else
			{
				matrix_shower=new Matrix_shower(viewer_frame,mtr_data,null,true);
			}

			matrix_shower.setTitle("Map of delays in communications for all processors");
			matrix_shower.setVisible(true);
		}
		else
		{
			matrix_shower.update(mtr_data,mtr_deviation);
		}
		

		diagram_panels_list = new ArrayList<Diagram_panel>();
		

	}

	public void update_messages_slider()
	{
		int min=data.get_min_read_message_length();
		int max=data.get_max_read_message_length();

		if((max==-1)||(min==-1))
		{
			return;
		}
		//System.out.printf("min=%d, max=%d\n",min,max);
		
		slider.setEnabled(false);
		
		slider.setMinimum(min);
		slider.setMaximum(max);
		slider.setValue(min);
		
		Dictionary<Integer,JLabel> dictionary= new Hashtable<Integer,JLabel>();
		Integer mes_len[]=data.get_messages_length();
		for(int i=0;i < mes_len.length;i++)
		{
			JLabel label= new JLabel(mes_len[i].toString());
			dictionary.put(mes_len[i],label);
		}

		slider.setLabelTable(dictionary);
		slider.setEnabled(true);
		slider.repaint();

		return;
	}

	public void set_local_normalization(boolean norm)
	{
		if(norm==true)
		{
			matrix_shower.set_local_normalization(true);
		}
		else
		{
			double min=data.get_min_data_value();
			double max=data.get_max_data_value();
			
			matrix_shower.set_min(min);
			matrix_shower.set_max(max);
			matrix_shower.set_local_normalization(false);

		}
	}

	public static void main(String[] args)
	{
		Network_view_window window = new Network_view_window(null,null,null);
	}
	
	
}

